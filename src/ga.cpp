// GA operators: initialization, selection, crossover, mutation, local search
#include "ga.h"

#include <omp.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

using namespace std;

namespace pdc {

namespace {

constexpr double   kEps   = 1e-9;                 // minimum gain that counts as improvement
constexpr uint64_t kGolden = 0x9E3779B97F4A7C15ULL;

// Per-thread scratch used while building children
struct ThreadScratch {
    LSScratch    ls;
    vector<char> used;
    vector<int>  pos1;
    vector<int>  pos2;
    vector<int>  dirty;
};

// Hilbert curve index of (x, y) on a 2^16 x 2^16 grid
uint64_t hilbert_key(uint32_t x, uint32_t y) {
    constexpr uint32_t N = 1u << 16;
    uint64_t d = 0;
    for (uint32_t s = N / 2; s > 0; s /= 2) {
        const uint32_t rx = (x & s) ? 1u : 0u;
        const uint32_t ry = (y & s) ? 1u : 0u;
        d += static_cast<uint64_t>(s) * s * ((3u * rx) ^ ry);
        if (ry == 0) {
            if (rx == 1) {
                x = N - 1 - x;
                y = N - 1 - y;
            }
            swap(x, y);
        }
    }
    return d;
}

// Tour built by sorting cities along a randomly shifted / flipped Hilbert curve.
// Every call with a different RNG state gives a different (but good) tour.
void space_filling_tour(Genome& g, const TSPInstance& inst,
                        double minx, double miny, double span,
                        mt19937_64& rng) {
    const int n = inst.n;
    uniform_real_distribution<double> uni(0.0, 1.0);
    const double sx = uni(rng), sy = uni(rng);
    const bool flip_x = rng() & 1, flip_y = rng() & 1, swap_xy = rng() & 1;

    vector<pair<uint64_t, int>> keys(static_cast<size_t>(n));
    for (int c = 0; c < n; ++c) {
        double u = span > 0 ? (inst.pts[c].x - minx) / span : 0.0;
        double v = span > 0 ? (inst.pts[c].y - miny) / span : 0.0;
        u += sx;  u -= floor(u);
        v += sy;  v -= floor(v);
        if (flip_x) u = 1.0 - u;
        if (flip_y) v = 1.0 - v;
        if (swap_xy) swap(u, v);
        const uint32_t xi = min<uint32_t>(65535u, static_cast<uint32_t>(u * 65536.0));
        const uint32_t yi = min<uint32_t>(65535u, static_cast<uint32_t>(v * 65536.0));
        keys[c] = {hilbert_key(xi, yi), c};
    }
    sort(keys.begin(), keys.end());

    g.resize(static_cast<size_t>(n));
    for (int k = 0; k < n; ++k) g[k] = keys[k].second;
}

// True if u and v are neighbours in the tour described by `pos`
inline bool adjacent(const int* pos, int n, int u, int v) {
    int d = pos[u] - pos[v];
    if (d < 0) d = -d;
    return d == 1 || d == n - 1;
}

}  // namespace

uint64_t splitmix64(uint64_t x) {
    x += kGolden;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// ---------------------------------------------------------------------------
// Local search: 2-opt + neighbour lists + work queue
// ---------------------------------------------------------------------------
//
// For city a and each tour neighbour b (successor or predecessor) we look at
// the K nearest cities c of a. Replacing edges (a,b),(c,d) by (a,c),(b,d)
// gains d(a,b)+d(c,d)-d(a,c)-d(b,d). Because the candidates c are sorted by
// distance, once d(a,c) >= d(a,b) no further c can help, so the scan stops.
// The tour is an array; a move reverses the shorter of the two paths.
double local_search_2opt(Genome& tour,
                         const TSPInstance& inst,
                         LSScratch& s,
                         const int* seeds,
                         int num_seeds,
                         bool use_oropt) {
    const int n = static_cast<int>(tour.size());
    const int K = inst.k;
    if (n < 5 || K == 0) return 0.0;

    s.pos.resize(static_cast<size_t>(n));
    s.queue.resize(static_cast<size_t>(n));
    s.inq.assign(static_cast<size_t>(n), 0);

    int*  t   = tour.data();
    int*  pos = s.pos.data();
    int*  q   = s.queue.data();
    char* inq = s.inq.data();

    for (int k = 0; k < n; ++k) pos[t[k]] = k;

    int qh = 0, qt = 0, qn = 0;   // ring buffer: head, tail, count (<= n)
    auto push = [&](int c) {
        if (!inq[c]) {
            inq[c] = 1;
            q[qt] = c;
            if (++qt == n) qt = 0;
            ++qn;
        }
    };

    if (seeds) {
        for (int i = 0; i < num_seeds; ++i) push(seeds[i]);
    } else {
        for (int k = 0; k < n; ++k) push(t[k]);
    }

    auto succ = [&](int c) { int p = pos[c] + 1; return t[p == n ? 0 : p]; };
    auto pred = [&](int c) { int p = pos[c];     return t[p == 0 ? n - 1 : p - 1]; };

    // Reverse the cyclic forward path from position i to position j. If that
    // path is longer than half the tour, reverse the complement instead (same
    // cyclic tour, mirrored), which keeps each move at most N/2 swaps.
    auto reverse_path = [&](int i, int j) {
        int len = j - i;
        if (len < 0) len += n;
        len += 1;
        if (2 * len > n) {
            const int ni = (j + 1 == n) ? 0 : j + 1;
            const int nj = (i == 0) ? n - 1 : i - 1;
            i = ni;
            j = nj;
            len = n - len;
        }
        for (int m = len / 2; m > 0; --m) {
            const int ci = t[i], cj = t[j];
            t[i] = cj;  pos[cj] = i;
            t[j] = ci;  pos[ci] = j;
            if (++i == n) i = 0;
            if (--j < 0) j = n - 1;
        }
    };

    // 2-opt move on two tour edges that are traversed in the same direction:
    // (a -> b) and (c -> d) become (a, c) and (b, d). The direction may be the
    // array direction or its mirror; that is detected from the current array.
    auto move2 = [&](int a, int b, int c, int d) {
        if (succ(a) == b) reverse_path(pos[b], pos[c]);
        else              reverse_path(pos[a], pos[d]);
    };

    // Or-opt: move a segment of 1..3 cities starting at s1 (read in either
    // direction) between two other neighbouring cities, reversed or not.
    // Executed as two or three 2-opt moves, so it reuses the same array
    // machinery and keeps the tour valid by construction.
    auto try_oropt = [&](int s1, double& out_gain) -> bool {
        for (int dir = 0; dir < 2; ++dir) {
            auto nxt = [&](int c) { return dir == 0 ? succ(c) : pred(c); };
            auto prv = [&](int c) { return dir == 0 ? pred(c) : succ(c); };

            const int p = prv(s1);
            int s2 = s1;
            for (int L = 1; L <= 3; ++L) {
                if (L > 1) s2 = nxt(s2);
                const int nx = nxt(s2);
                if (s2 == p || nx == p) break;
                const int mid = (L == 3) ? nxt(s1) : s1;

                const double g_rem = inst.dist(p, s1) + inst.dist(s2, nx) - inst.dist(p, nx);
                if (g_rem <= kEps) continue;

                const int* nb = inst.neighbor_row(s1);
                for (int m = 0; m < K; ++m) {
                    const int    c  = nb[m];
                    const double dc = inst.dist(s1, c);
                    if (dc >= g_rem - kEps) break;
                    if (c == s2 || c == mid) continue;

                    // A: ... c [s1..s2] d ...   (segment keeps its direction)
                    if (c != p) {
                        const int    d     = nxt(c);
                        const double added = dc + inst.dist(s2, d) - inst.dist(c, d);
                        if (g_rem - added > kEps) {
                            move2(p, s1, c, d);
                            move2(p, c, nx, s2);
                            move2(c, s2, s1, d);
                            out_gain = g_rem - added;
                            push(p); push(nx); push(s1); push(s2); push(c); push(d);
                            return true;
                        }
                    }
                    // B: ... e [s2..s1] c ...   (segment reversed)
                    if (c != nx) {
                        const int    e     = prv(c);
                        const double added = inst.dist(e, s2) + dc - inst.dist(e, c);
                        if (g_rem - added > kEps) {
                            move2(p, s1, e, c);
                            move2(p, e, nx, s2);
                            out_gain = g_rem - added;
                            push(p); push(nx); push(s1); push(s2); push(e); push(c);
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    };

    double gain = 0.0;

    while (qn > 0) {
        const int a = q[qh];
        if (++qh == n) qh = 0;
        --qn;
        inq[a] = 0;

        bool improved = true;
        while (improved) {
            improved = false;
            const int* nb = inst.neighbor_row(a);

            for (int dir = 0; dir < 2 && !improved; ++dir) {
                const int    b   = (dir == 0) ? succ(a) : pred(a);
                const double dab = inst.dist(a, b);

                for (int m = 0; m < K; ++m) {
                    const int    c  = nb[m];
                    const double g1 = dab - inst.dist(a, c);
                    if (g1 <= kEps) break;

                    const int d = (dir == 0) ? succ(c) : pred(c);
                    if (c == b || d == a) continue;

                    const double delta = g1 + inst.dist(c, d) - inst.dist(b, d);
                    if (delta > kEps) {
                        if (dir == 0) reverse_path(pos[b], pos[c]);
                        else          reverse_path(pos[a], pos[d]);
                        gain += delta;
                        push(b);
                        push(c);
                        push(d);
                        improved = true;
                        break;
                    }
                }
            }

            if (!improved && use_oropt && n >= 8) {
                double g = 0.0;
                if (try_oropt(a, g)) {
                    gain += g;
                    improved = true;
                }
            }
        }
    }
    return gain;
}

// ---------------------------------------------------------------------------
// Population set-up
// ---------------------------------------------------------------------------

void initialize_population(vector<Individual>& pop,
                           const TSPInstance& inst,
                           InitMode mode,
                           bool apply_local_search,
                           uint64_t seed,
                           bool use_oropt) {
    const int n = inst.n;
    const ptrdiff_t P = static_cast<ptrdiff_t>(pop.size());

    // Bounding box for the space-filling curve
    double minx = numeric_limits<double>::max(), miny = minx;
    double maxx = numeric_limits<double>::lowest(), maxy = maxx;
    for (const Point& p : inst.pts) {
        minx = min(minx, p.x);  maxx = max(maxx, p.x);
        miny = min(miny, p.y);  maxy = max(maxy, p.y);
    }
    const double span = max(maxx - minx, maxy - miny);

    #pragma omp parallel
    {
        LSScratch ls;

        #pragma omp for schedule(dynamic, 1)
        for (ptrdiff_t i = 0; i < P; ++i) {
            mt19937_64 rng(splitmix64(seed ^ (static_cast<uint64_t>(i) * kGolden)));
            Individual& ind = pop[i];

            if (mode == InitMode::SpaceFilling) {
                space_filling_tour(ind.genome, inst, minx, miny, span, rng);
            } else {
                ind.genome.resize(static_cast<size_t>(n));
                iota(ind.genome.begin(), ind.genome.end(), 0);
                shuffle(ind.genome.begin(), ind.genome.end(), rng);
            }

            if (apply_local_search) {
                local_search_2opt(ind.genome, inst, ls, nullptr, 0, use_oropt);
            }
            ind.fitness = tour_length(ind.genome, inst);
        }
    }
}

void evaluate_population(vector<Individual>& pop, const TSPInstance& inst) {
    const ptrdiff_t n = static_cast<ptrdiff_t>(pop.size());

    #pragma omp parallel for schedule(static)
    for (ptrdiff_t i = 0; i < n; ++i) {
        pop[i].fitness = tour_length(pop[i].genome, inst);
    }
}

// ---------------------------------------------------------------------------
// Genetic operators
// ---------------------------------------------------------------------------

// Best of `tournament_size` random individuals. Returns an index, so the
// 20 KB genomes are never copied during selection.
int tournament_select(const vector<Individual>& pop,
                      int tournament_size,
                      mt19937_64& rng) {
    uniform_int_distribution<size_t> pick(0, pop.size() - 1);

    size_t best = pick(rng);
    for (int k = 1; k < tournament_size; ++k) {
        const size_t cand = pick(rng);
        if (pop[cand].fitness < pop[best].fitness) best = cand;
    }
    return static_cast<int>(best);
}

// OX1: copy a slice of p1, then fill the remaining slots with p2's cities in
// the order they appear in p2 starting after the slice. Two linear loops
// instead of a modulo per element.
void order_crossover(const Genome& p1, const Genome& p2,
                     Genome& child, mt19937_64& rng,
                     vector<char>& used) {
    const size_t n = p1.size();
    if (n < 2) { child = p1; return; }

    child.resize(n);
    used.assign(n, 0);

    uniform_int_distribution<size_t> pick_i(0, n - 2);
    const size_t i = pick_i(rng);
    uniform_int_distribution<size_t> pick_j(i + 1, n - 1);
    const size_t j = pick_j(rng);

    for (size_t k = i; k <= j; ++k) {
        child[k]    = p1[k];
        used[p1[k]] = 1;
    }

    size_t write = (j + 1 == n) ? 0 : j + 1;
    auto place = [&](int city) {
        if (used[city]) return;
        child[write] = city;
        if (++write == n) write = 0;
    };
    for (size_t idx = j + 1; idx < n; ++idx) place(p2[idx]);
    for (size_t idx = 0; idx <= j; ++idx)    place(p2[idx]);
}

void order_crossover(const Genome& p1, const Genome& p2,
                     Genome& child, mt19937_64& rng) {
    vector<char> used;
    order_crossover(p1, p2, child, rng, used);
}

// Reverse a random short segment. Per-gene swaps at 2% would rewrite ~100
// positions of a 5000-city tour; one short inversion is a small, 2-opt-shaped
// perturbation that the local search can repair or exploit.
void inversion_mutation(Genome& genome, mt19937_64& rng, int max_len) {
    const int n = static_cast<int>(genome.size());
    if (n < 3) return;

    const int hi  = max(2, min(max_len, n));
    uniform_int_distribution<int> pick_len(2, hi);
    const int len = pick_len(rng);

    uniform_int_distribution<int> pick_start(0, n - len);
    const int start = pick_start(rng);

    reverse(genome.begin() + start, genome.begin() + start + len);
}

// ---------------------------------------------------------------------------
// One generation
// ---------------------------------------------------------------------------
//
// Child i uses its own RNG stream derived from (base_seed, i). The outcome is
// therefore identical for any thread count and any OpenMP schedule, which
// makes speedup measurements compare exactly the same work.
void evolve_generation(vector<Individual>& pop,
                       vector<Individual>& next,
                       const GAConfig& cfg,
                       const TSPInstance& inst,
                       mt19937_64& rng) {
    const int pop_size    = static_cast<int>(pop.size());
    const int elite_count = min(cfg.elitism_count, pop_size);
    const int n           = inst.n;

    // Only the elites need to be in order
    partial_sort(pop.begin(), pop.begin() + elite_count, pop.end(),
                 [](const Individual& a, const Individual& b) {
                     return a.fitness < b.fitness;
                 });

    next.resize(static_cast<size_t>(pop_size));
    for (int e = 0; e < elite_count; ++e) next[e] = pop[e];

    const uint64_t base_seed = rng();

    #pragma omp parallel
    {
        ThreadScratch sc;
        if (cfg.local_search) {
            sc.pos1.resize(static_cast<size_t>(n));
            sc.pos2.resize(static_cast<size_t>(n));
        }

        #pragma omp for schedule(dynamic, 4)
        for (int i = elite_count; i < pop_size; ++i) {
            mt19937_64 local_rng(splitmix64(base_seed ^ (static_cast<uint64_t>(i) * kGolden)));
            uniform_real_distribution<double> coin(0.0, 1.0);

            const int ia = tournament_select(pop, cfg.tournament_size, local_rng);
            const int ib = tournament_select(pop, cfg.tournament_size, local_rng);
            const Individual& P1 = pop[ia];
            const Individual& P2 = pop[ib];
            Individual& child    = next[i];

            const bool do_cross = coin(local_rng) < cfg.crossover_rate;
            if (do_cross) {
                order_crossover(P1.genome, P2.genome, child.genome, local_rng, sc.used);
            } else {
                child.genome = P1.genome;   // reuses the buffer's capacity
            }

            const bool do_mut = coin(local_rng) < cfg.mutation_rate;
            if (do_mut) inversion_mutation(child.genome, local_rng);

            // An unchanged clone keeps its parent's fitness
            if (!do_cross && !do_mut) {
                child.fitness = P1.fitness;
                continue;
            }

            if (cfg.local_search) {
                // Cities that touch an edge present in neither parent are
                // where the child can be improved; start the local search there.
                const int* q1 = sc.pos1.data();
                const int* q2 = do_cross ? sc.pos2.data() : sc.pos1.data();
                for (int k = 0; k < n; ++k) {
                    sc.pos1[P1.genome[k]] = k;
                    if (do_cross) sc.pos2[P2.genome[k]] = k;
                }

                sc.dirty.clear();
                const int* g = child.genome.data();
                for (int k = 0; k < n; ++k) {
                    const int u = g[k];
                    const int v = g[(k + 1 == n) ? 0 : k + 1];
                    if (!adjacent(q1, n, u, v) && !adjacent(q2, n, u, v)) {
                        sc.dirty.push_back(u);
                        sc.dirty.push_back(v);
                    }
                }

                if (!sc.dirty.empty()) {
                    local_search_2opt(child.genome, inst, sc.ls,
                                      sc.dirty.data(),
                                      static_cast<int>(sc.dirty.size()),
                                      cfg.use_oropt);
                }
            }

            child.fitness = tour_length(child.genome, inst);
        }
    }

    pop.swap(next);
}

}  // namespace pdc