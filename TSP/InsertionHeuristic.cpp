#include<bits/stdc++.h>

using namespace std;
int n, dist[105][105];
int min_dist[105];

long long Operation(int u) {
    vector<int> V, T;
    for (int i = 0; i < n; i++) V.push_back(i), min_dist[i] = dist[i][u];
    T.push_back(u);
    V.erase(find(V.begin(), V.end(), u));

    int k = u;
    for (int i = 0; i < n; i++) if (min_dist[i] < min_dist[k]) k = i;
    T.push_back(k);
    V.erase(find(V.begin(), V.end(), k));

    while (T.size() < n) {
        int k = V[0];
        for (auto x : V) if (min_dist[x] < min_dist[k]) k = x;
        for (auto x : V) min_dist[x] = min(min_dist[x], dist[x][k]);

        int pos = 0;
        long long min_diff_dist = 1e18; 
        for (int i = 0; i < T.size() - 1; i++) {
            if (dist[T[i]][k] + dist[k][T[i + 1]] - dist[T[i]][T[i + 1]] < min_diff_dist) {
                min_diff_dist = dist[T[i]][k] + dist[k][T[i + 1]] - dist[T[i]][T[i + 1]];
                pos = i;
            }
        }
        T.insert(T.begin() + pos + 1, k);
        V.erase(find(V.begin(), V.end(), k));
    }

    long long res = dist[T[n - 1]][T[0]];
    for (int i = 0; i < n - 1; i++) res += dist[T[i]][T[i + 1]];
    return res;
}

int main() {
#define task "a"
    freopen(task".inp", "r", stdin);
    // freopen(task".out", "w", stdout);

    cin >> n;
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) cin >> dist[i][j];
    for (int i = 0; i < n; i++) dist[i][i] = 1e9;

    long long res = 1e18;
    for (int i = 0; i < n; i++) res = min(res, Operation(i));
    cout << res;

    return 0;
}