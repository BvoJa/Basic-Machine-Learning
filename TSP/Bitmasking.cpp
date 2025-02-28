#include<bits/stdc++.h>

using namespace std;
int n, dist[22][22];
long long dp[22][1 << 22];

long long DP(int last, int mask, int sta) {
    long long &res = dp[last][mask];
    if (~res) return res;
    if (mask == 0) return res = dist[last][sta];

    long long sum = 1e18;
    for (int sub = mask, u; sub > 0; sub ^= (1 << u)) {
        u = __builtin_ctz(sub);
        sum = min(sum, DP(u, mask ^ (1 << u), sta) + dist[last][u]);
    }
    return res = sum;
} 

int main() {
#define task "a"
    freopen(task".inp", "r", stdin);
    // freopen(task".out", "w", stdout);

    cin >> n;
    for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) cin >> dist[i][j];
    
    memset(dp, -1, sizeof dp);  
    long long res = 1e18;
    for (int i = 0; i < n - 1; i++) res = min(res, DP(i, (1 << n) - 1 - (1 << i), i));
    cout << res;

    return 0;
}