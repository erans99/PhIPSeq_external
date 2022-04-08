import glob
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy
import pandas

import PhIPSeq_external.config as config
from PhIPSeq_external.Analyse_Fastq.map_reads_custom_primers import run_map_custom, get_lib_len
from PhIPSeq_external.Analyse_Fastq.score_serum_vs_base import run_serum_vs_input_levels

path_phage = config.PATH_PHAGE

MIN_BASE = 25
PVAL = 0.05
PLOT_LEVEL = 1
MAX_READS = 1.25 * 10 ** 6
MIN_PROP_READS = 0.6
OUT_DUP = True  # whether to create p_values for duplicates or only for one copy

LIBS = ["A", "T", "AT", "AC1", "AC2", "C2"]  # Agilent, Twist, Agilent&Twist, Agilent&Corona1, Agilent&Corona2, Corona2


def run_well(inp, out_base, ID, lib, cols, max_reads):
    print("Working on %s" % inp)
    out_path = os.path.join(out_base, ID)
    if os.path.exists(out_path):
        print("WTF, output path exists")
    else:
        os.makedirs(out_path)

    f_serum = os.path.join(out_path, "found_%s.csv" % ID)
    if not os.path.isfile(f_serum):
        # def run_map_custom(out_str, allow_indel, use5, f_input, out_dir, libname, sum_cols=[], max_good_reads=-1,
        #                    ext="", ext2=None)

        num_vars, num_reads = run_map_custom("", True, False, inp, out_path, lib, cols, max_reads,
                                             ext="_R1_001.fastq", ext2="_R2_001.fastq")
    else:
        try:
            num_vars = get_lib_len(lib)
            print("Found %s" % f_serum)
            df = pandas.read_csv(f_serum, index_col=0)
            num_reads = 0
            for c in cols:
                num_reads += df[c].sum()
        except:
            num_vars, num_reads = run_map_custom("", True, False, inp, out_path, lib, cols, max_reads,
                                                 ext="_R1_001.fastq", ext2="_R2_001.fastq")

    print("%s got %d good reads" % (os.path.basename(inp), num_reads))
    f_input_levels = os.path.join(config.BASE_PATH, "PE_input_levels", "%s", "found_all.csv") % lib
    out_path = os.path.join(out_path, ("res_%s_vs_%s_%s" % (os.path.splitext(os.path.basename(f_serum))[0],
                                                            os.path.splitext(os.path.basename(f_input_levels))[0],
                                                            "__".join(cols))))

    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    else:
        print("Output %s exits" % out_path)
    if len(glob.glob(os.path.join(out_path, "top_samp*.csv"))) != 0:
        return num_vars, out_path, num_reads, [None, None], None

    pr_out = open(os.path.join(out_path, "log.txt"), "w")
    lam_p, tet_b = run_serum_vs_input_levels(f_input_levels, f_serum, cols, 1., num_vars, out_path, PLOT_LEVEL,
                                             pr_out=pr_out)
    pr_out.close()
    return num_vars, out_path, num_reads, lam_p, tet_b


def set_name(df, base):
    df['order'] = df.index.astype(str)
    df['order'] = base + "_" + df['order']


def get_names(libname):
    if libname.upper() == 'T':
        df_all = pandas.read_csv(os.path.join(path_phage, "final_twist_with_info.csv"))
        set_name(df_all, 'twist')
    elif libname.upper() == 'A':
        df_all = pandas.read_csv(os.path.join(path_phage, "final_agilent_with_info.csv"))
        set_name(df_all, 'agilent')
    elif libname.upper() == 'AT':
        df_all = [pandas.read_csv(os.path.join(path_phage, "final_twist_with_info.csv")),
                  pandas.read_csv(os.path.join(path_phage, "final_agilent_with_info.csv"))]
        set_name(df_all[0], 'twist')
        set_name(df_all[1], 'agilent')
        df_all = pandas.concat(df_all, ignore_index=True)
    elif libname.upper() == 'AC1':
        df_all = [pandas.read_csv(os.path.join(path_phage, "final_corona1_with_info.csv")),
                  pandas.read_csv(os.path.join(path_phage, "final_agilent_with_info.csv"))]
        set_name(df_all[0], 'corona1')
        set_name(df_all[1], 'agilent')
        df_all = pandas.concat(df_all, ignore_index=True)
    elif libname.upper() == 'AC2':
        df_all = [pandas.read_csv(os.path.join(path_phage, "final_corona2_with_info.csv")),
                  pandas.read_csv(os.path.join(path_phage, "final_agilent_with_info.csv"))]
        set_name(df_all[0], 'corona2')
        set_name(df_all[1], 'agilent')
        df_all = pandas.concat(df_all, ignore_index=True)
    elif libname.upper() == 'C2':
        df_all = pandas.read_csv(os.path.join(path_phage, "final_corona2_with_info.csv"))
        set_name(df_all, 'corona2')
    else:
        print("Unknown lib!!! Exiting.")
        sys.exit(0)
    return df_all[['nuc_seq', 'pos', 'len_seq', 'full name', 'file', 'order']]


def fill_names(df, all_lib):
    out = all_lib.merge(df, 'outer', left_on='nuc_seq', right_index=True)
    out.set_index('nuc_seq', inplace=True)
    return out


def correct_and_out(tkttores, UID, ID, min_p, all_lib, bad_olis, plot=True):
    fs = glob.glob(os.path.join(tkttores[ID][1], "top*"))
    if len(fs) == 1:
        f = fs[0]
    else:
        print("WTF %s has %d files" % (os.path.join(tkttores[ID][1], "top*"), len(fs)))
        return
    df = pandas.read_csv(f, index_col=1)
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', 1, inplace=True)
    else:
        print("%s already corrected" % tkttores[ID][1])
        return
    df['fold'] = df['final_cnt'] / df['orig_cnt']
    df = fill_names(df, all_lib)
    df.fillna(0, inplace=True)
    df.loc[df.index.intersection(bad_olis), '-log10_p'] = numpy.nan
    if plot:
        plt.figure()
        plt.plot(range(len(df)), df['-log10_p'], linewidth=0.2)
        plt.title("P_val of %s (%d passed bonferroni)" % (UID, len(df[df['-log10_p'] > min_p])))
        ticks = map(lambda x: x[:-3] + 'k' if x[-3:] == '000' else x, df.iloc[range(0, len(df), 10000)]['order'].values)
        plt.xticks(range(0, len(df), 10000), list(ticks), rotation='vertical', fontsize='xx-small')
        plt.plot([0, len(df)], [min_p, min_p], color='r', linewidth=0.5)
        fig = plt.gcf()
        fig.set_size_inches((8.5, 6), forward=False)
        fig.savefig(os.path.join(os.path.dirname(f), "all_scores" + UID + ".png"), dpi=300)
        plt.close('all')
    df.sort_values('-log10_p', inplace=True, ascending=False)
    df.to_csv(f)
    return


def read_file(f, cols):
    found = pandas.read_csv(f, index_col=0)
    found['good'] = 0
    for c in cols:
        found['good'] += found[c]
    return found[['good']]


def check_anchor(tkttores, log, th, bad_olis, name, cmp_file, cols):
    bad_special = [0, 0]
    cmp_df = read_file(cmp_file, ['no_err'])
    cmp_df.drop(cmp_df.index.intersection(bad_olis), inplace=True)
    col = "good"
    cmp_tops = set(cmp_df[col].sort_values().index[-500:])
    bad_p = []
    for k in tkttores.keys():
        if k.split("_")[2] == name:
            df = read_file(glob.glob(os.path.join(os.path.dirname(tkttores[k][1]), "found*.csv"))[0], cols)
            df.drop(df.index.intersection(bad_olis), inplace=True)
            tops = set(df[col].sort_values().index[-100:])
            diff = tops.difference(cmp_tops)
            if len(diff) > 0:
                log += "%s has %d of top 100 oligos not in top 500 of Anchor pool\n" % \
                       (os.path.basename(tkttores[k][1]), len(diff))
            r = df[[col]].merge(cmp_df[[col]], 'outer', left_index=True, right_index=True)
            r = r.append(pandas.DataFrame(index=range(344000 - len(bad_olis) - len(r))), sort=False)
            r.fillna(0, inplace=True)
            res = r.corr(method='pearson').loc[col + '_x', col + '_y']
            if res < th:
                bad_special[0] += 1
                print(k, res)
                bad_p.append(res)
            bad_special[1] += 1
    if bad_special[0] == 0:
        print("All %d %s files are good (>= %g pearson)" % (bad_special[1], name, th))
        log += "All %d %s files are good (>= %g pearson)\n" % (bad_special[1], name, th)
    else:
        print("%d of %d %s files are bad (< %g pearson)" % (bad_special[0], bad_special[1], name, th))
        log += "%d of %d %s files are bad (< %g pearson)\n" % (bad_special[0], bad_special[1], name, th)
        log += str(bad_p)
    return [name] + bad_special, log


def check_mock(tkttores, log, max_p, name, col):
    bad_special = [0, 0, 0]
    bad_miss = 0
    bad_oligos = {}
    cnt = 0
    for k in tkttores.keys():
        if k.split("_")[2] == name:
            try:
                df = pandas.read_csv(glob.glob(os.path.join(tkttores[k][1], "top*.csv"))[0], index_col=0)
            except:
                print("No output for %s" % tkttores[k][1])
                bad_miss += 1
                continue
            cnt += 1
            if len(df[df[col] > max_p]) > 0:
                bad_special[0] += 1
                bad = df[df[col] > max_p].oligo.values
                for b in bad:
                    if b in bad_oligos.keys():
                        bad_oligos[b] += 1
                    else:
                        bad_oligos[b] = 1
                bad_special[1] += len(bad)
            bad_special[2] += 1
    if bad_miss > 0:
        print("%d mock files missing" % bad_miss)
        log += "%d mock files missing\n" % bad_miss
    if bad_special[0] == 0:
        print("All %d %s files are good (all oligos got %s < %g)" % (bad_special[2], name, col, max_p))
        log += "All %d %s files are good (all oligos got %s < %g)\n" % (bad_special[2], name, col, max_p)
    else:
        print("%d of %d %s files are bad (alltogether >%d oligos got %s < %g)" % (bad_special[0], bad_special[2],
                                                                                  name, bad_special[1], col, max_p))
        log += "%d of %d %s files are bad (alltogether >%d oligos got %s < %g)\n" % (bad_special[0], bad_special[2],
                                                                                     name, bad_special[1], col, max_p)
    for b in bad_oligos.keys():
        if bad_oligos[b] > 1:
            log += "%s appears at -10log(p_val) > %g for %d of %d mocks\n" % (b, max_p, bad_oligos[b], bad_special[2])
    bad_oligos = pandas.Series(bad_oligos)
    bad_olis = bad_oligos[bad_oligos >= (cnt / 2)].index
    print("%d olis will not be scorred" % len(bad_olis))
    log += "%d olis will not be scorred\n" % len(bad_olis)
    return [name] + bad_special, bad_olis, log


def check_negative_controls(tkttores, log, max_num, name, cols):
    col = 'good'
    bad_special = [0, 0]
    for k in tkttores.keys():
        if k.split("_")[2] == name:
            df = read_file(glob.glob(os.path.join(os.path.dirname(tkttores[k][1]), "found*.csv"))[0], cols)
            if len(df[df[col] > 0]) > max_num:
                bad_special[0] += 1
            bad_special[1] += 1
    if bad_special[0] == 0:
        print("All %d %s files are good (<=%d oligos)" % (bad_special[1], name, max_num))
        log += "All %d %s files are good (<=%d oligos)\n" % (bad_special[1], name, max_num)
    else:
        print("%d of %d %s files are bad (>%d oligos)" % (bad_special[0], bad_special[1], name, max_num))
        log += "%d of %d %s files are bad (>%d oligos)\n" % (bad_special[0], bad_special[1], name, max_num)
    return [name] + bad_special, log


def get_SampleSheet(fname):
    ls = open(fname).read().split("\n")
    found = [False, -1]
    for i, l in enumerate(ls):
        if ("SampleID" in l) and ("SampleName" in l):
            if not found[0]:
                found = [True, i]
            else:
                print("Can't parse SampleSheet. Format change? Exiting")
                sys.exit(1)

    if not found[0]:
        print("Can't parse SampleSheet. Format change? Exiting")
        sys.exit(1)
    samp_df = pandas.read_csv(fname, index_col=0, skiprows=found[1])
    if len(samp_df) < 96:
        print("Sample sheet has only %d lines. Please verify.")
    return samp_df


def choose_ID(IDs, tkttores, log):
    dat = {}
    for ID in IDs:
        dat[ID] = [tkttores[ID][2], tkttores[ID][3][0], tkttores[ID][3][1], tkttores[ID][4]]
    dat = pandas.DataFrame(dat, index=['num_reads', 'lam_intercept', 'lam_slope', 'theta']).T
    log += "From:\n"
    log += str(dat) + "\nChose:\n"
    dat = dat[dat.num_reads == dat.num_reads.max()]
    if len(dat) == 1:
        log += dat.index[0] + " because of num reads\n"
        return dat.index[0], log
    tmp_dat = dat[dat.lam_slope > 0]
    if len(tmp_dat) > 0:
        dat = tmp_dat
    if len(dat) == 1:
        log += dat.index[0] + " because of lambda slope\n"
        return dat.index[0], log
    tmp_dat = dat[dat.theta > 0]
    if len(tmp_dat) > 0:
        dat = tmp_dat
    if len(dat) == 1:
        log += dat.index[0] + " because of theta\n"
        return dat.index[0], log
    tmp_dat = dat[dat.lam_intercept > 0]
    if len(tmp_dat) > 0:
        dat = tmp_dat
    if len(dat) == 1:
        log += dat.index[0] + " because of lambda intercept\n"
        return dat.index[0], log
    log += dat.index[0] + " randomly\n"
    return dat.index[0], log


def remove_dir(path):
    com = "rm -rf %s" % path
    os.system(com)


def check_and_correct(tkttores, comments, IDs, out_path, cols, lib, max_reads, num_in_ref):
    log = ""
    for k in tkttores.keys():
        if tkttores[k][4] is not None:
            log += "%s found %d good reads (lam=%g+%gx theta=%g)\n" % (k, tkttores[k][2], tkttores[k][3][0],
                                                                       tkttores[k][3][1], tkttores[k][4])
        else:
            log += "%s found %d good reads (failed to estimate lambda and theta)\n" % (k, tkttores[k][2])
    num_vars = tkttores[list(tkttores.keys())[0]][0]
    min_p = -math.log10(PVAL / num_vars)
    print("Checking NCs (no phage, no serum)")
    _, log = check_negative_controls(tkttores, log, 10 ** 4, "NC", cols)
    print("Checking mocks (with phage, no serum)")
    _, bad_olis, log = check_mock(tkttores, log, min_p, "Mock", "-log10_p")
    print("Checking anchors (with phage, with joint three person serum)")
    if os.path.exists(os.path.join(config.BASE_PATH, "PE_anchor_levels", "%s", "found_all.csv") % lib):
        _, log = check_anchor(tkttores, log, 0.9, bad_olis, "Anchor",
                              os.path.join(config.BASE_PATH, "PE_anchor_levels", "%s", "found_all.csv") %
                              lib, cols)
    else:
        print("Can't check anchors. No base levels.")
        log += "Can't check anchors. No base levels.\n"

    out_path = os.path.join(out_path, "final_res")
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    all_lib = get_names(lib)
    res = {}
    for UID in IDs.keys():
        if UID.split('_')[1] in ["NC", "Mock", "Anchor"]:
            continue
        if True:
            if len(IDs[UID]) > 1:
                ID, log = choose_ID(IDs[UID], tkttores, log)
                for i in IDs[UID]:
                    if i == ID:
                        continue
                    if OUT_DUP:
                        correct_and_out(tkttores, UID, i, min_p, all_lib, bad_olis)
                    else:
                        remove_dir(tkttores[i][1])
            else:
                ID = IDs[UID][0]
            if tkttores[ID][2] < (MIN_PROP_READS * max_reads):
                print("%s has too few reads (%d). Can't output passed oligos." % (ID, tkttores[ID][2]))
                log += "%s has too few reads (%d). Can't output passed oligos.\n" % (ID, tkttores[ID][2])
                remove_dir(tkttores[ID][1])
                continue
            # remove oligos that appear too often in the mock
            # Cut only oligos with pvalue above min_p
            correct_and_out(tkttores, UID, ID, min_p, all_lib, bad_olis)
            try:
                f = glob.glob(os.path.join(tkttores[ID][1], "top*.csv"))[0]
                fold_norm = (num_in_ref / tkttores[ID][2])
                df = pandas.read_csv(f, index_col=0)
                pr = ""
                res[UID] = []
                pr += "%d " % len(df[df['-log10_p'] > min_p])
                res[UID].append(tkttores[ID][2])
                res[UID].append(comments[ID])
                res[UID].append(tkttores[ID][3] + [tkttores[ID][4]])
                res[UID].append(len(df[df['-log10_p'] > min_p]))
                tmp = df[(df['-log10_p'] > min_p) | (numpy.isnan(df['-log10_p']))].copy()
                tmp['fold'] = (tmp['final_cnt'] / tmp['orig_cnt'].apply(lambda x: max(x, MIN_BASE))) * fold_norm
                tmp.sort_values('-log10_p', inplace=True)
                tmp.to_csv(os.path.join(out_path, 'passed_oligo_scores_%s.csv' % UID))
            except:
                print("%s failed" % UID)
                if UID in res.keys():
                    res.pop(UID)
    res = pandas.DataFrame(res, index=['num_reads', 'comment', 'params', 'p_val']).T
    res.to_csv(os.path.join(out_path, "all_info.csv"))
    open(os.path.join(out_path, "all_log.txt"), "w").write(log)


def clean_dir(base_path, out_path, seq_num, plate, max_reads, ext, lib, rm=True):
    res_path = os.path.join(base_path, "all_%s_results" % lib)
    new_plate = plate + ("S%d" % seq_num)
    fs = glob.glob(os.path.join(out_path, "final_res", "passed*.csv"))
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    num_out = len(fs)

    for f in fs:
        com = "cp %s %s" % (f, os.path.join(res_path, os.path.basename(f).replace("_" + plate + "_",
                                                                                  "_" + new_plate + "_")))
        os.system(com)
    for f in ['all_info.csv', 'all_log.txt']:
        fs = glob.glob(os.path.join(out_path, "final_res", f))
        com = "cp %s %s" % (fs[0], os.path.join(res_path, f.replace("all", "%s_%d_%s" % (new_plate, max_reads, ext))))
        os.system(com)
    if rm:
        com = "rm -rf %s" % out_path
        os.system(com)
    return num_out


def run_plate(in_dir, seq_num, plate, max_reads, lib, num_in_ref, base_path=config.BASE_PATH, out_dir=config.BASE_PATH,
              **kwargs):
    cols = ['no_err', 'one_err', 'indel']
    ext = "all3"

    print("Running on: %s %sS%d" % (in_dir, plate, seq_num))
    print("%d reads of %s" % (max_reads, cols))

    print("Initializing")
    print("Starting")

    out_path = os.path.join(os.path.join(out_dir, "runsPE_%d", "%s", "%s_%s") % (max_reads, lib, plate, ext))
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    fs = glob.glob(os.path.join(in_dir, "Unaligned_fastq", "Sample_%s_*" % plate))
    SampSheet = get_SampleSheet(os.path.join(in_dir, "SampleSheet.csv"))

    IDs = {}
    tkttores = {}
    comments = {}

    for f in fs:
        try:
            ID = SampSheet.loc[os.path.basename(f)]['SampleName']
            com = SampSheet.loc[os.path.basename(f)]['Description']
        except:
            print("%s not in SampleSheet, or has no SampleName. Exiting" % os.path.basename(f))
            sys.exit(1)
        run_plate_name = ID.split('_')[0]
        UID = run_plate_name + "_" + ID.split("_")[2]
        comments[ID] = com
        tkttores[ID] = run_well(f, out_path, ID, lib, cols, max_reads)
        if UID in IDs.keys():
            IDs[UID].append(ID)
        else:
            IDs[UID] = [ID]
    if len(tkttores) == 0:
        print("Something wrong. Got no files. Please correct parameters and re-run")
        sys.exit(1)
    check_and_correct(tkttores, comments, IDs, out_path, cols, lib, max_reads, num_in_ref)

    num_out = clean_dir(base_path, out_path, seq_num, plate, max_reads, ext, lib, False)

    print("Done")
    return num_out


if __name__ == '__main__':
    if sys.version[0] != '3':
        print("This program works only on python3!!!")
        sys.exit(0)

    if len(sys.argv) < 3:
        print(f"{config.PYTHON_PATH} run_plate_NextSeq_PE.py " +
              "<in_dir> <num_seq_run> <plate> <lib> <max_reads>")
        print("<in_dir> name of NextSeq input directory (the FastqOutput directory)")
        print("<num_seq_run> number to be added to plate number. Please be carefull with this")
        print("<plate> plate in input directory, like R2P1")
        print("<lib> which library to run against options are %s. Default is AT (Agilent&Twist)" % (os.path.join(LIBS)))
        print("<max_reads> optional, maximal number of reads used. Default is %d" % MAX_READS)
        sys.exit(0)

    if len(sys.argv) >= 6:
        max_reads = int(sys.argv[5])
    else:
        max_reads = MAX_READS
    if len(sys.argv) >= 5:
        lib = sys.argv[4]
        if lib not in LIBS:
            print("Lib of %s is not valid. Should be one of" % lib, LIBS)
            sys.exit(0)
    else:
        lib = "AT"
    print("Running on %s %d %s lib %s %d" % (sys.argv[1], int(sys.argv[2]), sys.argv[3], lib, max_reads))

    fnum = os.path.join(config.BASE_PATH, "PE_input_levels", str(lib), "num_all.csv")
    if os.path.exists(fnum):
        num_in_ref = int(open(fnum, "r").readline().strip())
    else:
        print("Can't find file %s. Exiting" % fnum)
        sys.exit(0)

    num_out = run_plate(sys.argv[1], int(sys.argv[2]), sys.argv[3], max_reads, lib, num_in_ref)
    print("Run end, for %d barcodes. Please check results and then export what has passed" % num_out)
