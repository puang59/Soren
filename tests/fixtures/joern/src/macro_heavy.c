int macro_heavy(struct ctx *c, char *buf, int n)
{
#ifdef FEATURE_NOT_BUILT
    int os_style = -1;
    if (n > 4) {
        os_style = buf[0];
        copy_name(c, buf, n);
    }
    if (os_style < 0)
        report(c, "unknown");
    c->style = os_style;
    flush(c);
#endif
    return 0;
}
