int while_loop(int n)
{
    int s = 0;
    while (n > 0) {
        s += n;
        n--;
    }
    return s;
}
