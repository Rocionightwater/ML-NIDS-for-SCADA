#!/bin/bash

CMD="python preprocess-data.py 
-d ./IanArffDataset.arff
--split 0.6 0.2 0.2
--encode-function"

TIMESERIES_DATASETS_DIR=time-series-datasets
NORMAL_DATASETS_DIR=normal-datsets

${CMD} --label binary --time-series -n mean --payload-kmeans-imputed 3 3 -o ${TIMESERIES_DATASETS_DIR}/binary-ts-mean-kmeans
${CMD} --label binary --time-series -n mean --payload-gmm-imputed 2 6 -o ${TIMESERIES_DATASETS_DIR}/binary-ts-mean-gmm
${CMD} --label binary --time-series -n mean --payload-keep-value-imputed -o ${TIMESERIES_DATASETS_DIR}/binary-ts-mean-keep
${CMD} --label binary --time-series -n mean --payload-indicator-imputed -o ${TIMESERIES_DATASETS_DIR}/binary-ts-mean-indi

${CMD} --label binary --time-series -n minmax --payload-kmeans-imputed 3 3 -o ${TIMESERIES_DATASETS_DIR}/binary-ts-minmax-kmeans
${CMD} --label binary --time-series -n minmax --payload-gmm-imputed 2 6 -o ${TIMESERIES_DATASETS_DIR}/binary-ts-minmax-gmm
${CMD} --label binary --time-series -n minmax --payload-keep-value-imputed -o ${TIMESERIES_DATASETS_DIR}/binary-ts-minmax-keep
${CMD} --label binary --time-series -n minmax --payload-indicator-imputed -o ${TIMESERIES_DATASETS_DIR}/binary-ts-minmax-indi

${CMD} --label binary -n mean --payload-kmeans-imputed 3 3 -o ${NORMAL_DATASETS_DIR}/binary-std-mean-kmeans
${CMD} --label binary -n mean --payload-gmm-imputed 2 6 -o ${NORMAL_DATASETS_DIR}/binary-std-mean-gmm
${CMD} --label binary -n mean --payload-keep-value-imputed -o ${NORMAL_DATASETS_DIR}/binary-std-mean-keep
${CMD} --label binary -n mean --payload-indicator-imputed -o ${NORMAL_DATASETS_DIR}/binary-std-mean-indi

${CMD} --label binary -n minmax --payload-kmeans-imputed 3 3 -o ${NORMAL_DATASETS_DIR}/binary-std-minmax-kmeans
${CMD} --label binary -n minmax --payload-gmm-imputed 2 6 -o ${NORMAL_DATASETS_DIR}/binary-std-minmax-gmm
${CMD} --label binary -n minmax --payload-keep-value-imputed -o ${NORMAL_DATASETS_DIR}/binary-std-minmax-keep
${CMD} --label binary -n minmax --payload-indicator-imputed -o ${NORMAL_DATASETS_DIR}/binary-std-minmax-indi

${CMD} --label category --time-series -n mean --payload-kmeans-imputed 3 3 -o ${TIMESERIES_DATASETS_DIR}/category-ts-mean-kmeans
${CMD} --label category --time-series -n mean --payload-gmm-imputed 2 6 -o ${TIMESERIES_DATASETS_DIR}/category-ts-mean-gmm
${CMD} --label category --time-series -n mean --payload-keep-value-imputed -o ${TIMESERIES_DATASETS_DIR}/category-ts-mean-keep
${CMD} --label category --time-series -n mean --payload-indicator-imputed -o ${TIMESERIES_DATASETS_DIR}/category-ts-mean-indi

${CMD} --label category --time-series -n minmax --payload-kmeans-imputed 3 3 -o ${TIMESERIES_DATASETS_DIR}/category-ts-minmax-kmeans
${CMD} --label category --time-series -n minmax --payload-gmm-imputed 2 6 -o ${TIMESERIES_DATASETS_DIR}/category-ts-minmax-gmm
${CMD} --label category --time-series -n minmax --payload-keep-value-imputed -o ${TIMESERIES_DATASETS_DIR}/category-ts-minmax-keep
${CMD} --label category --time-series -n minmax --payload-indicator-imputed -o ${TIMESERIES_DATASETS_DIR}/category-ts-minmax-indi

${CMD} --label category -n mean --payload-kmeans-imputed 3 3 -o ${NORMAL_DATASETS_DIR}/category-std-mean-kmeans
${CMD} --label category -n mean --payload-gmm-imputed 2 6 -o ${NORMAL_DATASETS_DIR}/category-std-mean-gmm
${CMD} --label category -n mean --payload-keep-value-imputed -o ${NORMAL_DATASETS_DIR}/category-std-mean-keep
${CMD} --label category -n mean --payload-indicator-imputed -o ${NORMAL_DATASETS_DIR}/category-std-mean-indi

${CMD} --label category -n minmax --payload-kmeans-imputed 3 3 -o ${NORMAL_DATASETS_DIR}/category-std-minmax-kmeans
${CMD} --label category -n minmax --payload-gmm-imputed 2 6 -o ${NORMAL_DATASETS_DIR}/category-std-minmax-gmm
${CMD} --label category -n minmax --payload-keep-value-imputed -o ${NORMAL_DATASETS_DIR}/category-std-minmax-keep
${CMD} --label category -n minmax --payload-indicator-imputed -o ${NORMAL_DATASETS_DIR}/category-std-minmax-indi


