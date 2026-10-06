int switch_case(int k)
{
    int r = 0;
    switch (k) {
    case 0:
        r = 10;
        break;
    case 1:
        r = 20;
    default:
        r = 30;
    }
    return r;
}
