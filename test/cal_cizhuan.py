num_43181 = 119
num_32181 = 27 + 3
num_62157 = 125 + 25
num_35157 = 47 + 14
num_60908 = 62 + 4

area_43181 = 300 * 450 * (0.1 ** 6)
area_32181 = 300 * 300 * (0.1 ** 6)
area_62157 = 300 * 600 * (0.1 ** 6)
area_35157 = 300 * 300 * (0.1 ** 6)
area_60908 = 600 * 600 * (0.1 ** 6)

pingmi_43181 = area_43181 * num_43181
pingmi_32181 = area_32181 * num_32181
pingmi_62157 = area_62157 * num_62157
pingmi_35157 = area_35157 * num_35157
pingmi_60908 = area_60908 * num_60908

print "\n".join([":".join(map(str, [k, v])) for k, v in locals().items() if "pingmi" in k])
