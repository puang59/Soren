int nested_loops(int n, int m)
{
    int c = 0;
    for (int i = 0; i < n; i++) {
        while (m > 0) {
            if (c > 100)
                break;
            c += m;
            m--;
        }
    }
    return c;
}
