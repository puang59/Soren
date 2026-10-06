int early_return(char *p)
{
    if (p == 0)
        return -1;
    use(p);
    return 0;
}
