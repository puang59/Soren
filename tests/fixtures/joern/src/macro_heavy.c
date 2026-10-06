DEFINE_HANDLER(my_handler)
{
    BEGIN_TABLE(tbl)
    ENTRY(1, foo)
    ENTRY(2, bar)
    ENTRY(3, baz)
    ENTRY(4, qux)
    END_TABLE
    FOR_EACH_ITEM(it, tbl)
    PROCESS(it)
    END_FOR
    return 0;
}
