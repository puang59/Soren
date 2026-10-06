int multi_line(char *dst, const char *src, int n)
{
    int len = n;
    if (len > 0 &&
        dst != 0 &&
        src != 0) {
        memcpy(dst,
               src,
               len);
    }
    return check(dst,
                 len);
}
