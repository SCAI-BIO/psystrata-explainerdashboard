import random
import pandas
from sklearn.model_selection import train_test_split

psychatric_history_dict = {
    "psychiatrictreatmentsetting_current_v1": ["Inpatient", "Outpatient", "Daycare", "Other"],
    "hospitalforpsychiatricreasons_v1": ["Yes", "No", "Unknown"],
    "hospitalforpsychiatricreasons_times_v1": [0, 10],
    "dayshospital_history_v1": [0, 100],
    "hospitalizedreason_history_v1": ["psychosis-related", "depression-related", "bipolar depression- related",
                                      "suicidal ideation", "Other"],
    "Hospitalization_voluntary_involuntary_unknown": ["Voluntary", "Involuntary", "Unknown"],
    "hospitalizedinvolreason_history_v1": ["danger to self", "danger to others", "danger to self and others", "Other"],
}

sociodemographic_data_dict = {
    "genderidentity_v1": ["Male", "Female", "Non-binary", "Prefer not to say", "Other"],
    "sexatbirth_v1": ["Male", "Female"],
    "occupation_current_yesno_v1": ["Yes", "No"],
    "yearsinschool_v1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    "degree_participant_v1": [
        "Unknown", "Less than high school", "High school (unfinished)",
        "High school (finished)", "Professional training (unfinished)",
        "Professional training (finished)", "University (unfinished)",
        "University (finished)", "Other"
    ],
    "educationaldifficulties_v1": [
        "Never in special education, but did experience educational problems",
        "Held back 1 year, no special education",
        "Required special education but progressed at a rate of one grade per year and/or held back less than 2 years",
        "Held back 2 or more years",
        "Did not complete school",
        "No problems",
        "Unknown"
    ],
    "occupation_father_v1": [
        "High executive", "Major professional", "Administrative personnel",
        "Minor professional", "Owner small business", "Sales", "Technician",
        "Farmer", "Skilled manual employee", "Unskilled manual employee",
        "Student", "Homemaker (in household of at least two persons)",
        "Unpaid work/voluntary work", "Unknown"
    ],
    "degree_father_v1": [
        "Unknown", "Less than high school", "High school (unfinished)",
        "High school (finished)", "Professional training (unfinished)",
        "Professional training (finished)", "University (unfinished)",
        "University (finished)", "Other"
    ],
    "occupation_mother_v1": [
        "High executive", "Major professional", "Administrative personnel",
        "Minor professional", "Owner small business", "Sales", "Technician",
        "Farmer", "Skilled manual employee", "Unskilled manual employee",
        "Student", "Homemaker (in household of at least two persons)",
        "Unpaid work/voluntary work", "Unknown"
    ],
    "degree_mother_v1": [
        "Unknown", "Less than high school", "High school (unfinished)",
        "High school (finished)", "Professional training (unfinished)",
        "Professional training (finished)", "University (unfinished)",
        "University (finished)", "Other"
    ],
    "maritalstatus_participant_v1": [
        "Never married", "In a relationship",
        "Married/Common Law/de facto (two people who live together as partners)",
        "Separated but not divorced", "Divorced", "Widowed",
        "Unknown", "Other"
    ],
    "children_participant_v1": [0, 1, 2, 3, 4, 5],
    "livingsituation_current_v1": [
        "House/flat/apartment with family of origin", "Rented room",
        "Rented flat/house/apartment", "Owned flat/house/apartment",
        "Boarding house/hostel", "Supported residential home",
        "Homeless or couch surfing", "Unknown", "Other"
    ],
    "livewith_current_v1": [
        "Alone", "Mother/step-mother/foster mother", "Father/step-father/foster father",
        "Sibling(s)", "Partner", "Son(s)/Daughter(s)", "Friend(s)", "Housemate",
        "Grandparents or extended family", "Support workers", "Unknown", "Other"
    ],
    "incomesource_v1": [
        "Contributions from family/spouse/partner", "Disability income",
        "Gross earnings (self)", "Social support", "Pension/Insurance",
        "Personal needs allowance (e.g., if living in a shelter)",
        "Student loan", "Unknown", "Other"
    ],
    "childhoodenvironment_v1": [
        "City (Population >500,000 people)",
        "City/Town (Population 100,000 - 500,000 people)",
        "City/Town (Population 10,000 - 100,000 people)",
        "Village/Rural", "Unknown"
    ],
    "occupation_current_v1": [
        "High executive", "Major professional", "Administrative personnel",
        "Minor professional", "Owner small business", "Sales", "Technician",
        "Farmer", "Skilled manual employee", "Unskilled manual employee",
        "Student", "Homemaker (in household of at least two persons)",
        "Unpaid work/voluntary work", "Unknown"
    ],
    "occupation_previous_yesno_v1": ["Yes", "No"],
    "occupation_previous_v1": [
        "High executive", "Major professional", "Administrative personnel",
        "Minor professional", "Owner small business", "Sales", "Technician",
        "Farmer", "Skilled manual employee", "Unskilled manual employee",
        "Student", "Homemaker (in household of at least two persons)",
        "Unpaid work/voluntary work", "Unknown"
    ]
}

psychiatric_medication_data_dict = {
    "Fluoxetine": [10, 20, 40, 60],
    "Sertraline": [25, 50, 100],
    "Escitalopram": [5, 10, 20],
    "Venlafaxine": [37.5, 75, 150, 225],
    "Duloxetine": [30, 60, 120],
    "Bupropion": [75, 100, 150, 300],
    "Risperidone": [0.25, 0.5, 1, 2, 4],
    "Olanzapine": [2.5, 5, 10, 15, 20],
    "Quetiapine": [25, 50, 100, 200, 300],
    "Aripiprazole": [5, 10, 15, 20, 30],
    "Clozapine": [25, 50, 100, 200],
    "Lithium": [300, 450, 600],
    "Valproate": [125, 250, 500],
    "Lamotrigine": [25, 50, 100, 200],
    "Carbamazepine": [200, 400],
    "Alprazolam": [0.25, 0.5, 1, 2],
    "Lorazepam": [0.5, 1, 2],
    "Diazepam": [2, 5, 10],
    "Buspirone": [5, 10, 15, 30],
    "Methylphenidate": [5, 10, 20, 36],
    "Amphetamine/Dextroamphetamine": [5, 10, 20, 30],
    "Lisdexamfetamine": [10, 20, 30, 40, 50, 60, 70],
    "Zolpidem": [5, 10],
    "Eszopiclone": [1, 2, 3],
    "Temazepam": [7.5, 15, 30],
    "Trazodone": [50, 100, 150, 300]
}


def gen_name():
    first_names = [
        "John", "Jane", "Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Hannah",
        "Isaac", "Julia", "Kevin", "Linda", "Mary", "Nathan", "Olivia", "Peter", "Quinn", "Rachel",
        "Steve", "Tina", "Ursula", "Victor", "Wendy", "Xavier", "Yvonne", "Zach", "Amber", "Brandon",
        "Catherine", "Derek", "Ella", "Frederick", "Gabriella", "Henry", "Ivy", "Jack", "Kara",
        "Liam", "Megan", "Noah", "Ophelia", "Patrick", "Rose", "Sophia", "Thomas", "Uma", "Violet",
        "Walter", "Xena", "Yasmine", "Zane", "Aaron", "Beth", "Chloe", "Dylan", "Eleanor", "Finn",
        "Georgia", "Harvey", "Isabel", "Jake", "Kelsey", "Lucas", "Mia", "Nina", "Oscar", "Paige",
        "Ryan", "Samantha", "Tyler", "Vanessa", "Willow", "Zoe"
    ]

    last_names = [
        "Smith", "Johnson", "Williams", "Jones", "Brown", "Davis", "Miller", "Wilson", "Moore", "Taylor",
        "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin", "Thompson", "Garcia", "Martinez",
        "Robinson", "Clark", "Rodriguez", "Lewis", "Lee", "Walker", "Hall", "Allen", "Young", "King",
        "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green", "Adams", "Nelson", "Baker",
        "Gonzalez", "Carter", "Mitchell", "Perez", "Roberts", "Turner", "Phillips", "Campbell", "Parker",
        "Evans", "Edwards", "Collins", "Stewart", "Sanchez", "Morris", "Rogers", "Reed", "Cook", "Morgan",
        "Bell", "Murphy", "Bailey", "Rivera", "Cooper", "Richardson", "Cox", "Howard", "Ward", "Torres",
        "Peterson", "Gray", "Ramirez", "James", "Watson", "Brooks", "Kelly", "Sanders", "Price", "Bennett"
    ]

    return random.choice(first_names) + " " + random.choice(last_names)


def gen_age():
    return random.randint(18, 100)


def gen_psychatric_history():
    psychatric_history = {}
    for key, value in psychatric_history_dict.items():
        psychatric_history[key] = random.choice(value)
    return psychatric_history


def gen_sociodemographic_data():
    sociodemographic_data = {}
    for key, value in sociodemographic_data_dict.items():
        sociodemographic_data[key] = random.choice(value)
    return sociodemographic_data


def gen_medication_with_resistance():
    medication = {}
    treatment_resistance_chance = 0  # Start with a low chance of treatment resistance

    # Medication-based logic: Certain medications are more associated with resistance
    for key, value in psychiatric_medication_data_dict.items():
        if random.random() < 0.9:
            medication[key] = 0
        else:
            dosage = random.choice(value)
            medication[key] = dosage

            # If medication is one associated with resistance, increase the chance
            if key in ["Olanzapine", "Clozapine", "Quetiapine"]:
                # scale resistance chance based on dosage
                treatment_resistance_chance += 0.1 * (dosage / 100)

    return medication, treatment_resistance_chance


def gen_patient() -> pandas.DataFrame:
    age = gen_age()
    medication, medication_resistance_chance = gen_medication_with_resistance()

    # Age-based logic: Older patients are more likely to have treatment resistance
    # scale medication resistance chance based on age
    medication_resistance_chance += 0.01 * (age - 50)

    # Determine treatment resistance: 0 or 1 (resistance or not)
    treatment_resistance = 1 if random.random() < min(1, medication_resistance_chance) else 0

    # Generate patient data
    patient = {
        "Name": gen_name(),
        "Age": age,
        **gen_psychatric_history(),
        **gen_sociodemographic_data(),
        **medication,
        "Treatment Resistance": treatment_resistance
    }
    return pandas.DataFrame(patient, index=[0])


def gen_patients(n_patients) -> pandas.DataFrame:
    patients = []
    for _ in range(n_patients):
        patients.append(gen_patient())
    return pandas.concat(patients)


def get_cat_columns_from_dict(data_dict):
    return [key for key, value in data_dict.items() if isinstance(value[0], str)]


def one_hot_encode_cats(patients: pandas.DataFrame, cat_columns):
    return pandas.get_dummies(patients, columns=cat_columns)


def patients_train_test_split(patients: pandas.DataFrame, test_size=0.2):
    # generate X_train, y_train, X_test, y_test
    # index should always be the patient name
    patients = patients.set_index("Name")
    # one-hot-encode categorical columns
    cat_cols = get_cat_columns_from_dict({**psychatric_history_dict, **sociodemographic_data_dict})
    patients = one_hot_encode_cats(patients, cat_cols)
    X = patients.drop(columns=["Treatment Resistance"])
    y = patients["Treatment Resistance"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X_train, y_train, X_test, y_test


def patient_train_test_names(patients: pandas.DataFrame, test_size=0.2):
    X_train, _, X_test, _ = patients_train_test_split(patients, test_size)
    return X_train.index, X_test.index