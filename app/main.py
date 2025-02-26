import shap
from sklearn.ensemble import RandomForestClassifier

from app.datagen import gen_patients, patients_train_test_split, patient_train_test_names
from explainerdashboard import ClassifierExplainer, ExplainerDashboard, ExplainerHub

feature_descriptions = {
    "Name": "The name of the patient",
    "Age": "The age of the patient",
    "psychiatrictreatmentsetting_current_v1": "The current psychiatric treatment setting (e.g., inpatient, outpatient, daycare)",
    "hospitalforpsychiatricreasons_v1": "Indicates if the patient was hospitalized for psychiatric reasons",
    "hospitalforpsychiatricreasons_times_v1": "The number of times the patient was hospitalized for psychiatric reasons",
    "dayshospital_history_v1": "The number of days spent in the hospital historically",
    "hospitalizedreason_history_v1": "The reason for the patient’s historical hospitalizations",
    "Hospitalization_voluntary_involuntary_unknown": "The nature of hospitalization (voluntary, involuntary, or unknown)",
    "hospitalizedinvolreason_history_v1": "Reason for involuntary hospitalization in the patient's history",
    "ADHD_current": "Indicates if the patient has current ADHD",
    "ASD_current": "Indicates if the patient has current ASD (Autism Spectrum Disorder)",
    "genderidentity_v1": "The gender identity of the patient",
    "sexatbirth_v1": "The biological sex assigned to the patient at birth",
    "occupation_current_yesno_v1": "Indicates if the patient has a current occupation",
    "yearsinschool_v1": "The number of years the patient spent in school",
    "degree_participant_v1": "The highest degree obtained by the patient",
    "educationaldifficulties_v1": "Indicates if the patient had educational difficulties",
    "occupation_father_v1": "The occupation of the patient’s father",
    "degree_father_v1": "The highest degree obtained by the patient’s father",
    "occupation_mother_v1": "The occupation of the patient’s mother",
    "degree_mother_v1": "The highest degree obtained by the patient’s mother",
    "maritalstatus_participant_v1": "The marital status of the patient",
    "children_participant_v1": "Indicates if the patient has children",
    "livingsituation_current_v1": "The current living situation of the patient",
    "livewith_current_v1": "Details about whom the patient currently lives with",
    "incomesource_v1": "The source of income for the patient",
    "childhoodenvironment_v1": "The patient's childhood environment description",
    "occupation_current_v1": "The current occupation of the patient",
    "occupation_previous_yesno_v1": "Indicates if the patient had a previous occupation",
    "occupation_previous_v1": "The previous occupation of the patient",
    "Fluoxetine": "The dosage of Fluoxetine prescribed to the patient",
    "Sertraline": "The dosage of Sertraline prescribed to the patient",
    "Escitalopram": "The dosage of Escitalopram prescribed to the patient",
    "Venlafaxine": "The dosage of Venlafaxine prescribed to the patient",
    "Duloxetine": "The dosage of Duloxetine prescribed to the patient",
    "Bupropion": "The dosage of Bupropion prescribed to the patient",
    "Risperidone": "The dosage of Risperidone prescribed to the patient",
    "Olanzapine": "The dosage of Olanzapine prescribed to the patient",
    "Quetiapine": "The dosage of Quetiapine prescribed to the patient",
    "Aripiprazole": "The dosage of Aripiprazole prescribed to the patient",
    "Clozapine": "The dosage of Clozapine prescribed to the patient",
    "Lithium": "The dosage of Lithium prescribed to the patient",
    "Valproate": "The dosage of Valproate prescribed to the patient",
    "Lamotrigine": "The dosage of Lamotrigine prescribed to the patient",
    "Carbamazepine": "The dosage of Carbamazepine prescribed to the patient",
    "Alprazolam": "The dosage of Alprazolam prescribed to the patient",
    "Lorazepam": "The dosage of Lorazepam prescribed to the patient",
    "Diazepam": "The dosage of Diazepam prescribed to the patient",
    "Buspirone": "The dosage of Buspirone prescribed to the patient",
    "Methylphenidate": "The dosage of Methylphenidate prescribed to the patient",
    "Amphetamine/Dextroamphetamine": "The dosage of Amphetamine/Dextroamphetamine prescribed to the patient",
    "Lisdexamfetamine": "The dosage of Lisdexamfetamine prescribed to the patient",
    "Zolpidem": "The dosage of Zolpidem prescribed to the patient",
    "Eszopiclone": "The dosage of Eszopiclone prescribed to the patient",
    "Temazepam": "The dosage of Temazepam prescribed to the patient",
    "Trazodone": "The dosage of Trazodone prescribed to the patient",
    "Treatment Resistance": "Indicates if the patient exhibits treatment resistance",
}

cats = ['psychiatrictreatmentsetting_current_v1',
        'hospitalforpsychiatricreasons_v1',
        'hospitalizedreason_history_v1',
        'Hospitalization_voluntary_involuntary_unknown',
        'hospitalizedinvolreason_history_v1',
        # 'ADHD_current',
        # 'ASD_current',
        'genderidentity_v1',
        'sexatbirth_v1',
        'occupation_current_yesno_v1',
        'degree_participant_v1',
        'educationaldifficulties_v1',
        'occupation_father_v1',
        'degree_father_v1',
        'occupation_mother_v1',
        'degree_mother_v1',
        'maritalstatus_participant_v1',
        # 'children_participant_v1',
        'livingsituation_current_v1',
        'livewith_current_v1',
        'incomesource_v1',
        'childhoodenvironment_v1',
        'occupation_current_v1',
        'occupation_previous_yesno_v1',
        'occupation_previous_v1'
        ]

patients = gen_patients(1000)
patients.to_csv("patients.csv", index=False)

# generate train and test data, index should always be the patient name

X_train, y_train, X_test, y_test = patients_train_test_split(patients)
train_names, test_names = patient_train_test_names(patients)
model = RandomForestClassifier(n_estimators=50, max_depth=5)
model.fit(X_train, y_train)

explainer = ClassifierExplainer(model, X_test, y_test,
                                descriptions=feature_descriptions,
                                cats=cats,
                                labels=['Non Resistant', 'Resistant'],
                                idxs=test_names,
                                index_name="Patient",
                                target="Treatment Resistance",
                                )

db1 = ExplainerDashboard(explainer,
                         title="Data Scientist Board",
                         description="A detailed dashboard to explain the treatment resistance model.",
                         decision_trees=False,
                         whatif=False,
                         hide_poweredby=True,
                         )

db2 = ExplainerDashboard(explainer,
                         title="Clinician Board",
                         description="A less detailed dashboard to explore treatment options based on the model.",
                         importances=False,
                         model_summary=False,
                         decision_trees=False,
                         shap_dependence=False,
                         shap_interaction=False,
                         whatif=True,
                         # Individual predictions components that are hidden
                         hide_predindexselector=True,
                         hide_pdp=True,
                         hide_contributiongraph=True,
                         # whatif components that are hidden
                         hide_whatifindexselector=True,
                         hide_whatifpdp=True,
                         hide_whatifcontributiongraph=True,
                         # hide the powered by ExplainerDashboard logo
                         hide_poweredby=True,
                         )

hub = ExplainerHub(dashboards=[db1, db2],
                   title='PsychSTRATA Decision Support Dashboard',
                   description='The following dashboard is a prototype for a decision Support system addressing '
                               'Treatment Resistance (TR) in psychiatric patients. The dashboard is designed evaluate '
                               'potential factors leading to TR based on a trained Machine Learning (ML) model. '
                               'All data that is currently displayed in the dashboard is fully randomly generated '
                               'based on data dictionaries and does not relate to or represent real patients. '
                               'The underlying model was trained on randomly sampled feature distributions - resulting '
                               'explanations and conclusion are not clinically valid.',
                   n_dashboard_cols=2,
                   url='https://psych-strata.eu/',
                   model_name='Treatment Resistance',
                   model_type='Random Forest Classifier',
                   )
hub.run()
