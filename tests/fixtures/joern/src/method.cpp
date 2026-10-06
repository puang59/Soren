int Parser::ReadByte(const char *buf, int pos)
{
    if (pos >= size_)
        return -1;
    return buf[pos];
}
