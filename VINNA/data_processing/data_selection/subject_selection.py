# IMPORTS
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn")


def selection(df_all, stratify, trainsize=25, testsize=15, valsize=5, drop=True):
    """
    Function to split data into three separate, stratified sets.
    :param pd.DataFrame df_all: dataframe with data (subject and variables to consider for split)
    :param list(str) stratify: columns of dataframe to split on (equal size in each portion)
    :param int trainsize: number of samples for training set
    :param int testsize: number of samples for testing set
    :param int valsize: number of samples for validation set
    :param bool drop: whether to drop Gender and Age from returned dataframe
    :return pd.DataFrame: Dataframe listing each subject and assigned Set (Training, Validation, Testing)
    """
    train, test = train_test_split(df_all, test_size=testsize, train_size=trainsize + valsize, random_state=33,
                                   stratify=df_all[stratify])
    train, val = train_test_split(train, test_size=valsize, train_size=trainsize, random_state=33, stratify=train[stratify])

    train["Set"] = "Training"
    val["Set"] = "Validation"
    test["Set"] = "Testing"
    all = pd.concat([train, val, test], axis=0)
    if drop:
        all.drop(["Gender", "Age"], axis=1, inplace=True)
    return all


def visualize_splits(df, strat_el, outfile):
    # Modify dataframe to include all split vars as one column
    df_melt = pd.melt(df, id_vars="Set", value_vars=strat_el, var_name="Strata", value_name="Strat_Value")
    df_melt["Strat_Value"] = df_melt["Strat_Value"].apply(lambda x: 1 if x == "female" else 2)

    # Actual plotting
    g = sns.FacetGrid(data=df_melt, col="Strata", hue="Set", sharex=False)
    g.map_dataframe(sns.kdeplot, "Strat_Value")
    g.add_legend(loc="upper center")
    g.savefig(outfile)


def visualize_boxplots(df, columns, by, saveas, seaborn=False):
    if seaborn:
        hue = by[1] if len(by) == 2 else None
        df = df.melt(id_vars=by, value_vars=columns,
                value_name="Strata Value", var_name="Strata")
        g = sns.catplot(data=df, x=by[0], y="Strata Value",
                           col="Strata", hue=hue, kind="box",
                           sharey=False, col_wrap=3)
    else:
        g = df.boxplot(column=columns, by=by,
                       return_type="axes")
        if by:
            for ax in g:
                ax.set_ylim(0, 50)
        else:
            g.set_ylim(0, 50)

    if saveas:
        g.savefig(saveas)
    plt.show()
    return g


def discretize_and_combine(meta_df, strat_col_con, strat_col_cat):
    # Discretize continous columns (split into three bins based on quartiles)
    for col in strat_col_con:
        meta_df["Quartile_" + col] = pd.qcut(meta_df[col], 2, labels=False) + 1
        # add discretized continous cols to categorical columns
        strat_col_cat.append("Quartile_" + col)
    return meta_df, strat_col_cat


def create_set_csv(df, set_type, prefix, midfix, outfile):
    val = df[df["Set"] == set_type]
    val["Prefix"] = prefix
    val["Midfix"] = midfix
    val["Fin"] = val.agg('{0[Prefix]}{0[participant_id]}{0[Midfix]}{0[session_id]}'.format, axis=1)
    val.to_csv(outfile, sep=",", columns=["Fin"], header=False, index=False)
    return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--meta", type=str, help="File with relevant meta-information.",
                        default="/autofs/vast/lzgroup/Users/LeonieHenschel/FastInfantSurfer/data/subject_meta.csv")
    parser.add_argument("-scat", "--strat_cat", action="append", type=str,
                        default=["sex"],
                        help="Categorical elements to stratify on. Can pass multiple (always add --s flag for each item)")
    parser.add_argument("-scon", "--strat_cont", action="append", type=str,
                        default=["birth_age", "scan_age"],
                        help="Continous elements to stratify on. Will be categorized before use. "
                             "Can pass multiple (always add --s flag for each item)")
    parser.add_argument("-o", "--out", type=str,
                        default="/autofs/vast/lzgroup/Users/LeonieHenschel/FastInfantSurfer/data/dataset_split.tsv",
                        help="Outputfile under which split sets should be saved.")
    parser.add_argument("-oh", "--out_hist", type=str, default="",
                        help="Create histogram of splits and save it under the given name.")
    parser.add_argument("-test", "--size_testset", type=int,
                        help="Desired size of testset.", default=25)
    parser.add_argument("-val", "--size_valset", type=int, default=18,
                        help="Desired size of validationset.")
    parser.add_argument("-train", "--size_trainset", type=int,
                        help="Desired size of testingset.", default=25)
    arguments = parser.parse_args()
    return arguments


if __name__ == "__main__":
    # get arguments
    args = parse_args()

    # read meta data
    seperator = "\t" if args.meta[-3] == "t" else ","
    meta = pd.read_csv(args.meta, sep=seperator)

    # Selection is only possible on discrete values,
    # birth_weight, birth_age, scan_age are continous
    # and need to be discretized first
    meta, strat_on = discretize_and_combine(meta, args.strat_cont, args.strat_cat)

    # Create selection set and write to file
    sel_sets = selection(meta, args.strat_cat,
                         trainsize=args.size_trainset,
                         testsize=args.size_testset,
                         valsize=args.size_valset, drop=False)
    sel_sets.to_csv(args.out, sep="\t", index=False)

    # separately write training, validation and testing set to file (just path = IDs)
    base="/autofs/vast/lzgroup/Projects/FastInfantSurfer/Data"
    create_set_csv(sel_sets, "Validation", base + "/sub-", "_ses-",
                   args.out[:-4] + "_validation.csv")
    create_set_csv(sel_sets, "Testing", base + "/sub-", "_ses-",
                   args.out[:-4] + "_testing.csv")
    create_set_csv(sel_sets, "Training", base + "/sub-", "_ses-",
                   args.out[:-4] + "_training.csv")

    visualize_boxplots(sel_sets, ["birth_age", "scan_age", "birth_weight"],
                       ["sex"], args.out[:-4] + "_box_split.svg", True)

    visualize_boxplots(sel_sets, ["birth_age", "scan_age", "birth_weight"],
                       ["sex", "Set"], args.out[:-4] + "_box_orig.svg", seaborn=True)

    # Optionally, create histogram to visualize splits
    if args.out_hist:
        visualize_splits(sel_sets, args.strat_cat + args.strat_cont, args.out_hist)
    """
    orig = pd.read_csv("/datasets/uc-davis-autism/meta.csv", sep="\t")
    print(orig.columns)
    df_test = selection(orig[["ID", "Gender"]], "Gender")
    df_test = pd.merge(orig, df_test, how="left", on="ID")
    df_test.rename({"Gender_x": "Gender"}, inplace=True)
    df_test.drop("Gender_y", inplace=True, axis=1)
    df_test.to_csv("/datasets/uc-davis-autism/meta.csv", sep="\t", index=False)
    
    # mindboggle
    mm = pd.read_csv("/mindboggle/subject_list.txt", names=["ID"])
    mm_test = selection(mm, None)
    mm_test.to_csv("/datasets/uc-davis-autism/mindboggle.tsv", sep="\t", index=False)
    
    # oasis1
    orig = pd.read_csv("/datasets/opencc_benchmark_full/analysis/meta_selection.csv", sep=",")
    orig = orig[orig["Age"] > 60]
    orig_ad = orig[orig["Diagnosis"] == "AD"]
    orig_hc = orig[orig["Diagnosis"] == "HC"]
    df_test_ad = selection(orig_ad[["ID", "Gender", "Age"]], ["Gender"],
                           trainsize=19, testsize=9, valsize=2)
    df_test_hc = selection(orig_hc[["ID", "Gender", "Age"]], ["Gender"],
                           trainsize=40, testsize=9, valsize=4)

    df_test = pd.concat([df_test_ad,
                         df_test_hc], axis=0)

    df_test = pd.merge(orig, df_test, how="left", on="ID")
    print(df_test["Age"].mean(), df_test[["Age", "Set", "Diagnosis"]].groupby(["Set", "Diagnosis"]).agg("mean"))
    df_test.to_csv("/datasets/opencc_benchmark_full/analysis/split_meta.tsv", sep="\t",
                   index=False, columns=["ID", "Diagnosis", "Gender", "Age", "Set"])
    
    # Harp
    orig = pd.read_csv("/datasets/HarP/top_bottom40_HarP.tsv", sep="\t")
    df_test = selection(orig, ["MaxDiff_Bin"], trainsize=50, testsize=20, valsize=10, drop=False)
    print(df_test["Age"].mean(), df_test[["Age", "Hippocampus", "Set", "Diagnosis", "MaxDiff_Bin"]].groupby(["Set", "MaxDiff_Bin"]).agg("mean"))

    print(df_test.groupby(["Set", "MaxDiff_Bin"]).mean())
    df_test.to_csv("/datasets/HarP/analysis/split_meta_HarP.tsv", sep="\t",
                   index=False, columns=["ID", "MaxDiff_Bin", "Dataset", "Hippocampus", "Diagnosis", "Gender", "Age", "Set"])
    """
