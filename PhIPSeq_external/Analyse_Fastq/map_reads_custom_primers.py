import glob
import gzip
import os
import pickle
import sys
import time

import pandas

import PhIPSeq_external.config as config

path_phage = config.PATH_PHAGE
path_dicts = config.PATH_DICTS

PL1 = -15
PL2 = -30
PL3 = -44
PL4 = -59
PL5 = -75
BAD = -1000

ADAP = False
ALLOW_HAM_DIST_START = 3

# load saved object from file
def load_obj(path, name):
    with open(os.path.join(path, (name + '.pkl')), 'rb') as f:
        return pickle.load(f)


def oppose_strand(read):
    #    trans = maketrans("ACGT", "TGCA")
    trans = {ord('A'): ord('T'), ord('T'): ord('A'), ord('C'): ord('G'), ord('G'): ord('C')}
    #    read = unicode(read)
    read = read.translate(trans)[::-1]
    return read


def read_1full_read(f, file_num, num_reads):
    global ADAP
    l = []

    for i in range(4):
        l.append(f.readline())
        if len(l[-1]) == 0:
            print("End of file %d, %d reads overall" % (file_num + 1, num_reads))
            return None
        if type(l[-1]) == bytes:
            l[-1] = l[-1].decode("utf-8")
    if (l[0][0] != '@') or (l[2][0] != '+') or (len(l[1]) != len(l[3])):
        print("error in file, %d reads overall" % num_reads)
        return None
    if (len(l[3]) < 75) and (not ADAP):
        print("Note: Irrelevant adapters may have been trimmed!!!")
        ADAP = True
    return l


def ham_dist(b1, b2, len_comp=-1):
    if len_comp == -1:
        len_comp = len(b1)
    if (len(b1) != len(b2)) or (len_comp > len(b1)):
        print("Lens don't match")
        return -1
    cnt = 0
    for i in range(len_comp):
        cnt += (b1[i] != b2[i])
    return cnt


def try_indel_1(b1, b2, len_comp=-1):
    if len_comp == -1:
        len_comp = len(b1)
    for i in range(len_comp):
        if b1[i] != b2[i]:
            break
    bad = False
    for j in range(i, len_comp):
        if b1[j] != b2[j + 1]:
            bad = True
            break
    if not bad:
        return [0, 1, 0]
    bad = False
    for j in range(i, len_comp - 1):
        if b1[j + 1] != b2[j]:
            bad = True
            break
    if not bad:
        return [0, 0, 1]
    return []


def find_in_lib(df_all, field, seq):
    tmp = df_all[df_all[field] == seq]
    if len(tmp) == 1:
        return tmp.index[0]
    else:
        return -1


pr_pe = 0


# if there is a sequencing of the beggining of the oligo - make sure its a match and has no indels.
def check_pe(seq, l2):
    global pr_pe
    if l2 is None:
        return None
    seq2 = l2[1][:-1]
    if seq2 == seq[:len(seq2)]:
        return 0
    ham = ham_dist(seq2, seq[:len(seq2)])
    if ham <= ALLOW_HAM_DIST_START:
        return ham
    return -1


def map_to_lib_w_indel(df_all, dicts, op, found, not_found, read, sum_cs, l2):
    if len(dicts) == 5:
        return map_to_lib5_w_indel(df_all, dicts, op, found, not_found, read, sum_cs, l2)
    else:
        return map_to_lib3_w_indel(df_all, dicts, op, found, not_found, read, sum_cs, l2)


def map_to_lib3_w_indel(df_all, dicts, op, found, not_found, read, sum_cs, l2):
    pls = [dicts['end0_len15'].get(op[PL1:]), \
           dicts['end1_len15'].get(op[PL2:PL1]), \
           dicts['end2_len14'].get(op[PL3:PL2])]
    if (len(set(pls)) == 1) and (pls[0] is not None):
        seq = df_all.loc[pls[0]].nuc_seq
        ham = check_pe(seq, l2)
        if ham == -1:
            return not_found + 1, 0, ham
        if seq in found.keys():
            found[seq][0] += 1
        else:
            found[seq] = [1, 0, 0]
        return not_found, (0 in sum_cs), ham
    for pl in pls:
        if (pl is not None) and (pls.count(pl) == 2):
            seq = df_all.loc[pl].nuc_seq
            ham = check_pe(seq, l2)
            if ham == -1:
                return not_found + 1, 0, ham
            if seq in found.keys():
                found[seq][1] += 1
            else:
                found[seq] = [0, 1, 0]
            return not_found, (1 in sum_cs), ham

    pls_sh1 = [dicts['end0_len15'].get(op[PL1 - 1:-1]), \
               dicts['end1_len15'].get(op[PL2 - 1:PL1 - 1]), \
               dicts['end2_len14'].get(op[PL3 - 1:PL2 - 1]), \
               dicts['end1_len15'].get(op[PL2 + 1:PL1 + 1]), \
               dicts['end2_len14'].get(op[PL3 + 1:PL2 + 1])]
    maybe = []
    if (pls[0] == pls_sh1[2]) and (pls[0] is not None):
        maybe.append(pls[0])
    if (pls[0] == pls_sh1[4]) and (pls[0] is not None):
        maybe.append(pls[0])
    if (pls_sh1[1] == pls_sh1[2]) and (pls_sh1[1] is not None):
        maybe.append(pls_sh1[1])
    if (pls_sh1[3] == pls_sh1[4]) and (pls_sh1[3] is not None):
        maybe.append(pls_sh1[3])
    if len(set(maybe)) == 1:
        seq = df_all.loc[maybe[0]].nuc_seq
        ham = check_pe(seq, l2)
        if ham == -1:
            return not_found + 1, 0, ham
        if seq in found.keys():
            found[seq][2] += 1
        else:
            found[seq] = [0, 0, 1]
        return not_found, (2 in sum_cs), ham
    return not_found + 1, 0, None


def map_to_lib5_w_indel(df_all, dicts, op, found, not_found, read, sum_cs, l2):
    min_correct = 3
    pls = [dicts['end0_len15'].get(op[PL1:]), \
           dicts['end1_len15'].get(op[PL2:PL1]), \
           dicts['end2_len14'].get(op[PL3:PL2]), \
           dicts['end3_len15'].get(op[PL4:PL3]), \
           dicts['end4_len16'].get(op[PL5:PL4])]
    if (len(set(pls)) == 1) and (pls[0] is not None):
        seq = df_all.loc[pls[0]].nuc_seq
        ham = check_pe(seq, l2)
        if ham == -1:
            return not_found + 1, 0, ham
        if seq in found.keys():
            found[seq][0] += 1
        else:
            found[seq] = [1, 0, 0, 0, 0]
        return not_found, (0 in sum_cs), ham
    for pl in pls:
        if (pl is not None) and (pls.count(pl) >= min_correct):
            try:
                seq = df_all.loc[pl].nuc_seq
            except:
                continue
            ham = check_pe(seq, l2)
            if ham == -1:
                return not_found + 1, 0, ham
            if seq in found.keys():
                found[seq][5 - pls.count(pl)] += 1
            else:
                if (pls.count(pl) == 4):
                    found[seq] = [0, 1, 0, 0, 0]
                else:
                    found[seq] = [0, 0, 1, 0, 0]
            to_sum = ((5 - pls.count(pl)) in sum_cs)
            return not_found, to_sum, ham

    pls_shf = [pls, \
               [dicts['end0_len15'].get(op[PL1 - 1:-1]), \
                dicts['end1_len15'].get(op[PL2 - 1:PL1 - 1]), \
                dicts['end2_len14'].get(op[PL3 - 1:PL2 - 1]), \
                dicts['end3_len15'].get(op[PL4 - 1:PL3 - 1]), \
                dicts['end4_len16'].get(op[PL5 - 1:PL4 - 1])], \
               [None, \
                dicts['end1_len15'].get(op[PL2 + 1:PL1 + 1]), \
                dicts['end2_len14'].get(op[PL3 + 1:PL2 + 1]), \
                dicts['end3_len15'].get(op[PL4 + 1:PL3 + 1]), \
                dicts['end4_len16'].get(op[PL5 + 1:PL4 + 1])], \
               [dicts['end0_len15'].get(op[PL1 - 2:-2]), \
                dicts['end1_len15'].get(op[PL2 - 2:PL1 - 2]), \
                dicts['end2_len14'].get(op[PL3 - 2:PL2 - 2]), \
                dicts['end3_len15'].get(op[PL4 - 2:PL3 - 2]), \
                dicts['end4_len16'].get(op[PL5 - 2:PL4 - 2])], \
               [None, \
                dicts['end1_len15'].get(op[PL2 + 2:PL1 + 2]), \
                dicts['end2_len14'].get(op[PL3 + 2:PL2 + 2]), \
                dicts['end3_len15'].get(op[PL4 + 2:PL3 + 2]), \
                dicts['end4_len16'].get(op[PL5 + 2:PL4 + 2])]]

    # search for single indel (maybe with also 1 error)
    maybe = []
    for sh_pos in range(4):
        # insert
        tmp_pls = pls_shf[0][:sh_pos] + pls_shf[1][sh_pos + 1:]
        for pl in tmp_pls:
            if (pl != None) and (tmp_pls.count(pl) >= min_correct):
                maybe.append(pl)
        # delete
        tmp_pls = pls_shf[0][:sh_pos] + pls_shf[2][sh_pos + 1:]
        for pl in tmp_pls:
            if (pl != None) and (tmp_pls.count(pl) >= min_correct):
                maybe.append(pl)

    if len(set(maybe)) == 1:
        seq = df_all.loc[maybe[0]].nuc_seq
        ham = check_pe(seq, l2)
        if ham == -1:
            return not_found + 1, 0, ham
        if seq in found.keys():
            found[seq][3] += 1
        else:
            found[seq] = [0, 0, 0, 1, 0]
        return not_found, (3 in sum_cs), ham
    elif len(set(maybe)) > 1:
        return not_found + 1, 0, None, True

    # search for double indel
    maybe = []
    for sh_pos1 in range(3):
        for sh_pos2 in range(sh_pos1 + 1, 4):
            # insert and delete
            tmp_pls = pls_shf[0][:sh_pos1] + pls_shf[1][sh_pos1 + 1:sh_pos2] + pls_shf[0][sh_pos2 + 1:]
            for pl in tmp_pls:
                if (pl is not None) and (tmp_pls.count(pl) >= min_correct):
                    maybe.append(pl)
            # two inserts
            tmp_pls = pls_shf[0][:sh_pos1] + pls_shf[1][sh_pos1 + 1:sh_pos2] + pls_shf[3][sh_pos2 + 1:]
            for pl in tmp_pls:
                if (pl is not None) and (tmp_pls.count(pl) >= min_correct):
                    maybe.append(pl)
            # delete and insert
            tmp_pls = pls_shf[0][:sh_pos1] + pls_shf[2][sh_pos1 + 1:sh_pos2] + pls_shf[0][sh_pos2 + 1:]
            for pl in tmp_pls:
                if (pl is not None) and (tmp_pls.count(pl) >= min_correct):
                    maybe.append(pl)
            # two deletes
            tmp_pls = pls_shf[0][:sh_pos1] + pls_shf[2][sh_pos1 + 1:sh_pos2] + pls_shf[4][sh_pos2 + 1:]
            for pl in tmp_pls:
                if (pl is not None) and (tmp_pls.count(pl) >= min_correct):
                    maybe.append(pl)

    if len(set(maybe)) == 1:
        #        print (maybe[0], pls, pls_sh1)
        #        print (read, sh, len(op))
        seq = df_all.loc[maybe[0]].nuc_seq
        ham = check_pe(seq, l2)
        if ham == -1:
            return not_found + 1, 0, ham
        if seq in found.keys():
            found[seq][4] += 1
        else:
            found[seq] = [0, 0, 0, 0, 1]
        return not_found, (4 in sum_cs), ham
    return not_found + 1, 0, None


def map_to_lib(df_all, dicts, op, found, not_found, read, sum_cs, l2):
    if len(dicts) == 5:
        return map_to_lib5(df_all, dicts, op, found, not_found, read, sum_cs, l2)
    else:
        return map_to_lib3(df_all, dicts, op, found, not_found, read, sum_cs, l2)


def map_to_lib3(df_all, dicts, op, found, not_found, read, sum_cs, l2):
    pls = [dicts['end0_len15'].get(op[PL1:]), \
           dicts['end1_len15'].get(op[PL2:PL1]), \
           dicts['end2_len14'].get(op[PL3:PL2])]
    if (len(set(pls)) == 1) and (pls[0] is not None):
        seq = df_all.loc[pls[0]].nuc_seq
        ham = check_pe(seq, l2)
        if ham == -1:
            return not_found + 1, 0, ham
        if seq in found.keys():
            found[seq][0] += 1
        else:
            found[seq] = [1, 0]
        return not_found, (0 in sum_cs), ham
    for pl in pls:
        if (pl is not None) and (pls.count(pl) == 2):
            seq = df_all.loc[pl].nuc_seq
            ham = check_pe(seq, l2)
            if ham == -1:
                return not_found + 1, 0, ham
            if seq in found.keys():
                found[seq][1] += 1
            else:
                found[seq] = [0, 1]
            return not_found, (1 in sum_cs), ham
    return not_found + 1, 0, None


def map_to_lib5(df_all, dicts, op, found, not_found, read, sum_cs, l2):
    pls = [dicts['end0_len15'].get(op[PL1:]), \
           dicts['end1_len15'].get(op[PL2:PL1]), \
           dicts['end2_len14'].get(op[PL3:PL2]), \
           dicts['end3_len15'].get(op[PL4:PL3]), \
           dicts['end4_len16'].get(op[PL5:PL4])]
    if (len(set(pls)) == 1) and (pls[0] is not None):
        seq = df_all.loc[pls[0]].nuc_seq
        ham = check_pe(seq, l2)
        if ham == -1:
            return not_found + 1, 0, ham
        if seq in found.keys():
            found[seq][0] += 1
        else:
            found[seq] = [1, 0, 0]
        return not_found, (0 in sum_cs), ham
    for pl in pls:
        if (pl is not None) and (pls.count(pl) >= 3):
            seq = df_all.loc[pl].nuc_seq
            ham = check_pe(seq, l2)
            if ham == -1:
                return not_found + 1, 0, ham
            if seq in found.keys():
                found[seq][5 - pls.count(pl)] += 1
            else:
                if (pls.count(pl) == 4):
                    found[seq] = [0, 1, 0]
                else:
                    found[seq] = [0, 0, 1]
            to_sum = ((5 - pls.count(pl)) in sum_cs)
            return not_found, to_sum, ham
    return not_found + 1, 0, None


def create_ind_dict(df_all, field):
    d = {}
    for i in df_all.index:
        d[df_all.loc[i][field]] = i
    return d


def get_lib_len(libname):
    if libname.upper() == 'T':
        return len(pandas.read_csv(os.path.join(path_phage, "final_twist_with_info.csv")))
    elif libname.upper() == 'A':
        return len(pandas.read_csv(os.path.join(path_phage, "final_agilent_with_info.csv")))
    elif libname.upper() == 'AT':
        return (len(pandas.read_csv(os.path.join(path_phage, "final_twist_with_info.csv"))) +
                len(pandas.read_csv(os.path.join(path_phage, "final_agilent_with_info.csv"))))
    elif libname.upper() == 'AC1':
        return (len(pandas.read_csv(os.path.join(path_phage, "final_corona1_with_info.csv"))) +
                len(pandas.read_csv(os.path.join(path_phage, "final_agilent_with_info.csv"))))
    elif libname.upper() == 'AC2':
        return (len(pandas.read_csv(os.path.join(path_phage, "final_corona2_with_info.csv"))) +
                len(pandas.read_csv(os.path.join(path_phage, "final_agilent_with_info.csv"))))
    elif libname.upper() == 'C2':
        return (len(pandas.read_csv(os.path.join(path_phage, "final_corona2_with_info.csv"))))
    elif libname[-7:] == 'raw.txt':
        tmp = {}
        lines = open(libname, "r").readlines()
        for i, l in enumerate(lines):
            op = l.strip()
            tmp[i] = [op, op[PL1:], op[PL2:PL1], op[PL3:PL2], op[PL4:PL3], op[PL5:PL4]]
        return len(tmp)
    else:
        print("No such library")
        sys.exit(0)


def get_dicts(libname, use5):
    if libname.upper() == 'T':
        df_all = pandas.read_csv(os.path.join(path_phage, "final_twist_with_info.csv"))
        if use5:
            dicts = load_obj(path_dicts, "T_5dicts")
        else:
            dicts = load_obj(path_dicts, "T_3dicts")
    elif libname.upper() == 'A':
        df_all = pandas.read_csv(os.path.join(path_phage, "final_agilent_with_info.csv"))
        if use5:
            dicts = load_obj(path_dicts, "A_5dicts")
        else:
            dicts = load_obj(path_dicts, "A_3dicts")
    elif libname.upper() == 'AT':
        df_all = [pandas.read_csv(os.path.join(path_phage, "final_twist_with_info.csv")), \
                  pandas.read_csv(os.path.join(path_phage, "final_agilent_with_info.csv"))]
        df_all = pandas.concat(df_all, ignore_index=True)
        if use5:
            dicts = load_obj(path_dicts, "AT_5dicts")
        else:
            dicts = load_obj(path_dicts, "AT_3dicts")
    elif libname.upper() == 'AC1':
        df_all = [pandas.read_csv(os.path.join(path_phage, "final_agilent_with_info.csv")),
                  pandas.read_csv(os.path.join(path_phage, "final_corona1_with_info.csv"))]
        df_all = pandas.concat(df_all, ignore_index=True)
        if use5:
            dicts = load_obj(path_dicts, "AC1_5dicts")
        else:
            dicts = load_obj(path_dicts, "AC1_3dicts")
    elif libname.upper() == 'AC2':
        df_all = [pandas.read_csv(os.path.join(path_phage, "final_agilent_with_info.csv")),
                  pandas.read_csv(os.path.join(path_phage, "final_corona2_with_info.csv"))]
        df_all = pandas.concat(df_all, ignore_index=True)
        if use5:
            dicts = load_obj(path_dicts, "AC2_5dicts")
        else:
            dicts = load_obj(path_dicts, "AC2_3dicts")
    elif libname.upper() == 'C2':
        df_all = pandas.read_csv(os.path.join(path_phage, "final_corona2_with_info.csv"))
        if use5:
            dicts = load_obj(path_dicts, "C2_5dicts")
        else:
            dicts = load_obj(path_dicts, "C2_3dicts")
    elif libname[-7:] == 'raw.txt':
        tmp = {}
        lines = open(libname, "r").readlines()
        for i, l in enumerate(lines):
            op = l.strip()
            tmp[i] = [op, op[PL1:], op[PL2:PL1], op[PL3:PL2], op[PL4:PL3], op[PL5:PL4]]
        df_all = pandas.DataFrame(tmp).T
        cols = ["nuc_seq", "end0_len15", "end1_len15", "end2_len14", "end3_len15", "end4_len16"]
        df_all.columns = cols
        for field in cols[1: 4 + 2 * use5]:
            if len(df_all) != len(df_all[field].value_counts()):
                print("Non uniqueness in %s. Exiting." % field)
                sys.exit(0)
        dicts = {}
        for field in cols[1: 4 + 2 * use5]:
            dicts[field] = create_ind_dict(df_all, field)
            print("created dict %s" % field, time.ctime())
    else:
        print("No such library")
        sys.exit(0)
    return df_all, dicts


def run_map_custom(out_str, allow_indel, use5, f_input, out_dir, libname, sum_cols=[], max_good_reads=-1, ext="",
                   ext2=None, pr_all=False):
    global pr_pe
    pr_pe = 0
    if use5:
        if allow_indel:
            found_cols = ['no_err', 'one_err', 'two_errs', 'indel', 'two_indel']
        else:
            found_cols = ['no_err', 'one_err', 'two_errs']
    else:
        if allow_indel:
            found_cols = ['no_err', 'one_err', 'indel']
        else:
            found_cols = ['no_err', 'one_err']
    sum_cs = []
    for c in set(sum_cols):
        if not (c in found_cols):
            print("No column %s" % c)
        else:
            sum_cs.append(found_cols.index(c))

    print("Start", time.ctime())
    out_str += "Start" + str(time.ctime()) + "\n"
    df_all, dicts = get_dicts(libname, use5)
    print("Read library of size %d" % len(df_all))
    out_str += "Read library of size %d\n" % len(df_all)

    if os.path.isfile(f_input):
        print("Working on single file")
        fs = [f_input]
    else:
        fs = glob.glob(os.path.join(f_input, "*%s*" % ext))
        print("Directory had %d files" % len(fs))

    found = {}
    not_found = 0
    cnt = [0, 0]
    cnt_pe_match = [[0] * (ALLOW_HAM_DIST_START+2), [0] * (ALLOW_HAM_DIST_START+2)]
    sum_found = 0
    max_reached = False
    for f in fs:
        if 'Undetermined' in f:
            print("File name %s contains 'Undetermined' skipping" % f)
            continue
        print("Working on %s" % f)
        if (not ext2 is None) and (ext != ""):
            f2 = f.replace(ext, ext2)
        else:
            f2 = None
            fin2 = None
            l2 = None
        if os.path.splitext(f)[1] == '.gz':
            print("working on zipped file")
            fin = gzip.open(f, "rb")
            if not f2 is None:
                fin2 = gzip.open(f2, "rb")
        else:
            print("working on unzipped file")
            fin = open(f, "r")
            if not f2 is None:
                fin2 = open(f2, "r")
        l2 = None
        while True:
            if cnt[1] == 1000:
                print("at 1000 found %d IDs not found %d reads" % (len(found), not_found))
                if len(found) == 0:
                    return 0, 0
            if pr_all:
                if (cnt[1] % 10000) == 0:
                    print("At %d reads, in file #%d (%d, %d)" % (cnt[1], cnt[0], len(found), not_found), time.ctime())
            l = read_1full_read(fin, cnt[0], cnt[1])
            if l is None:
                break
            if not f2 is None:
                l2 = read_1full_read(fin2, cnt[0], cnt[1])
            if l2 is None:
                break
            read = l[1][:-1]
            op = oppose_strand(read)
            if allow_indel:
                not_found, good, pe_match = map_to_lib_w_indel(df_all, dicts, op, found, not_found, read, sum_cs, l2)
            else:
                not_found, good, pe_match = map_to_lib(df_all, dicts, op, found, not_found, read, sum_cs, l2)
            cnt[1] += 1
            sum_found += good
            if pe_match is not None:
                lib_ID = 0
                cnt_pe_match[lib_ID][pe_match] += 1
            if sum_found == max_good_reads:
                print("Got to max %d good reads. Stopping." % max_good_reads)
                max_reached = True
                break
        cnt[0] += 1
        if max_reached:
            break
    if cnt[1] == 0:
        print("No data found. Check input directory")
        sys.exit(0)
    if sum_found != max_good_reads:
        print("Got only %d good reads." % sum_found)
        out_str += "Got only %d good reads.\n" % sum_found
    print("Finished %d reads in %f files (%d, %d)" % (cnt[1], cnt[0], len(found), not_found), time.ctime())
    out_str += "Finished %d reads in %f files (%d, %d)" % (cnt[1], cnt[0], len(found), not_found) + \
               str(time.ctime()) + "\n"
    print()
    print("%d %d part IDs not found" % (not_found, len(dicts)))
    out_str += "%d %d part IDs not found\n" % (not_found, len(dicts))
    if len(found) > 0:
        found = pandas.DataFrame(found).T
        if os.path.isdir(out_dir):
            print("Dir %s exists. Writing result to it" % out_dir)
        else:
            os.mkdir(out_dir)
        if use5:
            if allow_indel:
                found.columns = found_cols
                found.to_csv(os.path.join(out_dir, "found_%s.csv" % os.path.basename(out_dir)))
                print("%d found with no err, %d with 1 errs (4 of 5 correct), %d with 2 errs (3 of 5 correct)" %
                      (found.no_err.sum(), found.one_err.sum(), found.two_errs.sum()))
                out_str += "%d found with no err, %d with 1 errs (4 of 5 correct), %d with 2 errs (3 of 5 correct)\n" % \
                           (found.no_err.sum(), found.one_err.sum(), found.two_errs.sum())
                print(("           %d with 1 indel (3 of 5 correct, after shift), " +
                       "%d with 2 indels (3 of 5 correct, after 2 shifts)") % (found.indel.sum(),
                                                                               found.two_indel.sum()))
                out_str += ("           %d with 1 indel (3 of 5 correct, after shift), " +
                            "%d with 2 indels (3 of 5 correct, after 2 shifts)\n\n") % (found.indel.sum(),
                                                                                        found.two_indel.sum())
                print()
                print("Found %d different IDs (with any correctable error)" % len(found))
                out_str += "Found %d different IDs (with any correctable error)\n" % len(found)
            else:
                found.columns = found_cols
                found.to_csv(os.path.join(out_dir, "found_%s.csv" % os.path.basename(out_dir)))
                print("%d found with no err, %d with 1 errs (4 of 5 correct), %d with 2 errs (3 of 5 correct)" %
                      (found.no_err.sum(), found.one_err.sum(), found.two_errs.sum()))
                out_str += ("%d found with no err, %d with 1 errs (4 of 5 correct), " +
                            "%d with 2 errs (3 of 5 correct)\n\n") % (found.no_err.sum(), found.one_err.sum(),
                                                                      found.two_errs.sum())
                print()
                print("Found %d different IDs (with <=1 error, no indel)" % len(found))
                out_str += "Found %d different IDs (with <=1 error, no indel)\n" % len(found)
        else:
            if allow_indel:
                found.columns = found_cols
                found.to_csv(os.path.join(out_dir, "found_%s.csv" % os.path.basename(out_dir)))
                print(("%d found with no err, %d with error (2 of 3 correct), %d with indel (2 of 3 correct, " +
                       "after shifts)") % (found.no_err.sum(), found.one_err.sum(), found.indel.sum()))
                out_str += ("%d found with no err, %d with error (2 of 3 correct), %d with indel (2 of 3 correct, " +
                            "after shifts)\n\n") % (found.no_err.sum(), found.one_err.sum(), found.indel.sum())
                print()
                print("Found %d different IDs (with any correctable error)" % len(found))
                out_str += "Found %d different IDs (with any correctable error)\n" % len(found)
            else:
                found.columns = found_cols
                found.to_csv(os.path.join(out_dir, "found_%s.csv" % os.path.basename(out_dir)))
                print("%d found with no err, %d with error (2 of 3 correct parts)" % (found.no_err.sum(),
                                                                                      found.one_err.sum()))
                out_str += "%d found with no err, %d with error (2 of 3 correct parts)\n\n" % (found.no_err.sum(),
                                                                                               found.one_err.sum())
                print()
                print("Found %d different IDs (with <=1 error, no indel)" % len(found))
                out_str += "Found %d different IDs (with <=1 error, no indel)\n" % len(found)
        if f2 is not None:
            tmp_out = "Compared short end: %d perfect, " % cnt_pe_match[0][0]
            for i in range(1, len(cnt_pe_match[0]) - 1):
                tmp_out += "%d %d errs, " % (cnt_pe_match[0][i], i)
            tmp_out += "%d not matched" % cnt_pe_match[0][-1]
            print(tmp_out)
            out_str += tmp_out + "\n"
        open(os.path.join(out_dir, "found_%s.txt" % os.path.basename(out_dir)), "w").write(out_str)
    else:
        print("Nothing found!?!?!?")

    print("End")
    return len(df_all), sum_found
