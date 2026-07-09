# 난임 환자 임신 성공 여부 예측 모델 미니 프로젝트

# EDA 전처리 코드
# 핵심 코드 요약

# 여부(0/1) 컬럼
for col in binary_columns:

    if col in train.columns:
        train[col] = (
            pd.to_numeric(train[col], errors="coerce")
            .astype("Int64")
        )

    if col in test.columns:
        test[col] = (
            pd.to_numeric(test[col], errors="coerce")
            .astype("Int64")
        )

# 소수점 컬럼 -> int64
for col in numeric_columns:

    if col in train.columns:
        train[col] = (
            pd.to_numeric(train[col], errors="coerce")
            .astype("Int64")
        )

    if col in test.columns:
        test[col] = (
            pd.to_numeric(test[col], errors="coerce")
            .astype("Int64")
        )

# 횟수 컬럼 
for col in count_columns:

    if col in train.columns:
        train[col] = (
            train[col]
            .map(count_map)
            .astype("Int64")
        )

    if col in test.columns:
        test[col] = (
            test[col]
            .map(count_map)
            .astype("Int64")
        )
        
# 특정 시술 유형 Feature engineering
for df in [train, test]:

    # -------------------------
    # ICSI 포함 여부
    # -------------------------
    df["시술_ICSI포함"] = (
        df["특정 시술 유형"]
        .str.contains("ICSI", na=False)
        .astype("Int64")
    )

    # -------------------------
    # IVF 포함 여부
    # -------------------------
    df["시술_IVF포함"] = (
        df["특정 시술 유형"]
        .str.contains(r"\bIVF\b", regex=True, na=False)
        .astype("Int64")
    )

    # -------------------------
    # BLASTOCYST 포함 여부
    # -------------------------
    df["시술_BLASTOCYST포함"] = (
        df["특정 시술 유형"]
        .str.contains("BLASTOCYST", na=False)
        .astype("Int64")
    )

    # -------------------------
    # AH 포함 여부
    # -------------------------
    df["시술_AH포함"] = (
        df["특정 시술 유형"]
        .str.contains(r"\bAH\b", regex=True, na=False)
        .astype("Int64")
    )

    # -------------------------
    # 인공수정 방식 추출
    # -------------------------
    method = df["특정 시술 유형"].str.extract(
        r"(IUI|ICI|IVI|Generic DI)",
        expand=False
    )

    method_map = {
        "IUI": 0,
        "ICI": 1,
        "IVI": 2,
        "Generic DI": 3
    }

    df["시술_인공수정방식"] = (
        method
        .map(method_map)
        .astype("Int64")
    )

    # -------------------------
    # 사용된 시술 기법 개수
    # -------------------------

    technique_count = (
        df["특정 시술 유형"]
        .fillna("")
        .str.count(":")
        +
        df["특정 시술 유형"]
        .fillna("")
        .str.count("/")
        + 1
    )

    technique_count = technique_count.where(
        df["특정 시술 유형"].notna(),
        pd.NA
    )

    df["시술_기법조합수"] = (
        technique_count
        .astype("Int64")
    )
