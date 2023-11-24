import typing as _T
import plotly
import pandas as pd
import matplotlib as mpl
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid")


def create_fake_image(img_shape: _T.Sequence) -> _T.Sequence:
    from skimage.draw import polygon, disk
    img = np.zeros(img_shape, dtype=np.float)
    # fill polygon
    poly = np.array((
        (1, 1),
        (img_shape[0]/8, img_shape[1]/4),
        (img_shape[0]/4, img_shape[1]/4),
    ))
    rr, cc = polygon(r=poly[:, 0], c=poly[:, 1], shape=img.shape)

    img[rr, cc, :] = 1

    # fill circle
    rr, cc = disk(center=(img_shape[0]-img_shape[0]/3, img_shape[1]-img_shape[1]/3),
                  radius=img_shape[0]/4, shape=img.shape)
    img[rr, cc, :] = 1
    return img


def boxplot(df, save_as, y="Structures", x="DSC", hue="Net_type", col=None):

    if col is not None:
        fig = plt.figure(figsize=(12.27, 9))
        ax = sns.boxplot(y=y, x=x, data=df, hue=hue)
        plt.xticks(list(range(10, 100, 5)))
    else:
        ax = sns.catplot(x=x, y=y, hue=hue, col=col,
                         data=df, kind="box")
    #fig.savefig(save_as)
    plt.savefig(save_as)


def combine_dfs(file_list, measure, network_cols):
    df_list = []
    network = network_cols[0]
    for f in file_list:
        df = read_csv(f)
        # Drop potential duplicates
        df = df.drop_duplicates(subset=['SubjectName'], keep='last', ignore_index=True)
        name = os.path.basename(f)[len(measure)+1:-4]
        print(f"Read dataframe - Shape: {df.shape}, Name: {name}")
        
        # Define ctx- and wm-labels
        right_ctx = [col for col in df.columns if col[-2:] == "GM" and col[:len("Right")] == "Right"]
        left_ctx = [col for col in df.columns if col[-2:] == "GM" and col[:len("Left")] == "Left"]
        right_wm = [col for col in df.columns if col[-2:] == "WM" and col[:len("Right")] == "Right"]
        left_wm = [col for col in df.columns if col[-2:] == "WM" and col[:len("Left")] == "Left"]
        
        if network not in df.columns or df[network][0] != name:
                df[network] = name
        #if "Left-choroid-plexus" in df.columns:
            #df = df.drop(['Brain-Stem', 'CSF', 'Left-Inf-Lat-Vent', 'Left-choroid-plexus', 'Right-Inf-Lat-Vent', 'Right-choroid-plexus', 'WM-hypointensities'], axis=1)
        if "Average" not in df.columns:
            # Take average over all classes
            avg_struct = [col for col in df.columns if col not in network_cols]
            df["Average"] = df[avg_struct].median(axis=1)
            print("Calculates Average")
        if "Right-Cerebral-Cortex" not in df.columns and "Cerebral-Cortex" not in df.columns:
            df["Right-Cerebral-Cortex"] = df[right_ctx].median(axis=1)
            print("Right CTX Average")
        if "Left-Cerebral-Cortex" not in df.columns and "Cerebral-Cortex" not in df.columns:
            df["Left-Cerebral-Cortex"] = df[left_ctx].median(axis=1)
            print("Left CTX Average")
        if "Right-Cerebral-White-Matter" not in df.columns and "Cerebral-White-Matter" not in df.columns:
            df["Right-Cerebral-White-Matter"] = df[right_wm].median(axis=1)
            print("Right WM Average")
        if "Left-Cerebral-White-Matter" not in df.columns and "Cerebral-White-Matter" not in df.columns:
            df["Left-Cerebral-White-Matter"] = df[left_wm].median(axis=1)
            print("Left WM Average")
        if "Subcortical" not in df.columns or True:
            # Take average over subcortical classes
            exclude = network_cols + ["Left-Cerebral-Cortex", "Right-Cerebral-Cortex", "Average", 
                                      "Left-Cerebral-White-Matter", "Right-Cerebral-White-Matter",
                                      "Left-Thalamus_high_intensity_part_in_T2", 
                                      "Right-Thalamus_high_intensity_part_in_T2",
                                      "Left-Thalamus_low_intensity_part_in_T2",
                                      "Right-Thalamus_low_intensity_part_in_T2", "Subcortical"]
            exclude.append(right_ctx)
            exclude.append(left_ctx)
            exclude.append(right_wm)
            exclude.append(left_wm)
            subcort_struct = [col for col in df.columns if col not in exclude]

            df["Subcortical"] = df[subcort_struct].median(axis=1)
            print("Subcortical Average")
        if measure == "dice":
            print("DSC times 100")
            df.loc[:, ~df.columns.isin(network_cols)] *= 100
        df_list.append(melt_structures(df, val_name=measure, var_col=network_cols))
    return pd.concat(df_list, ignore_index=True)


def read_csv(file):
    separator = "\t" if file[-3:] == "tsv" else ","
    return pd.read_csv(file, sep=separator, index_col=False)


def melt_structures(df, val_name="DSC", var_col=["Net_type"]):
    value_cols = [col for col in df.columns if col not in var_col and not col.startswith("Unnamed")]
    return pd.melt(df, id_vars=var_col, value_vars=value_cols,
                   var_name="Structures", value_name=val_name)


if __name__ == "__main__":
    import glob
    base = "/autofs/vast/lzgroup/Projects/FastInfantSurfer/experiments/"
    save = "/autofs/vast/lzgroup/Users/LeonieHenschel/FastInfantSurfer/data/boxplot_meas.png"
    net_col = "SF"
    measure = {"DSC": "dice_", "VS": "vs_", "ASD": "surfaceAverageHausdorff_"}
    meas_fin = []
    for key, value in measure.items():
        pattern = f"*_net1/eval_metrics/{value}ValSet*.tsv"
        suffix = [f"FastSurferVINN_net1/eval_metrics/{value}ValSet_FastSurferVINN_coronal.tsv",
                  f"FastSurferCNN_orig_orig/eval_metrics/{value}ValSet_FastSurferCNN_orig_coronal.tsv"]
        meas_list = [base + model for model in suffix]#glob.glob(base + pattern)
        meas_fin.append(combine_dfs(meas_list, key, net_col))

    import functools as ft
    df_final = ft.reduce(lambda left, right: pd.merge(left, right, on=["SubjectName", net_col, "Structures"]), meas_fin)
    key_list = measure.keys()
    test=df_final[key_list]
    #meas_fin = pd.concat(meas_fin, axis=0, ignore_index=True)
    df_final = pd.melt(df_final, id_vars=["SubjectName", net_col, "Structures"], value_vars=measure.keys(), var_name="Metric", value_name="Value")

    ax = sns.catplot(x="Value", y="Structures", hue=net_col, col="Metric",
                     data=df_final, kind="box", sharex=False, width=10, height=12.27)
    plt.tight_layout()
    plt.savefig(save)


    #boxplot(df_final, x="Structures", y="Value", hue=net_col, col="Metric", save_as=save)

    """
    test = "/autofs/vast/lzgroup/Projects/FastInfantSurfer/experiments/FastSurferVINN_net1/eval_metrics/dice_ValSet_FastSurferVINN_coronal.tsv"
    dsc = read_csv(test)

    # plot with plotly
    dsc_m = melt_structures(dsc)

    import plotly.express as px
    fig = px.box(dsc_m, x="DSC", y="Structures")
    fig.show()
    """