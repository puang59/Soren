int goto_out(int a)
{
    int r = -1;
    if (a < 0)
        goto out;
    r = work(a);
out:
    cleanup();
    return r;
}
