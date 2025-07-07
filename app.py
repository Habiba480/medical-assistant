import pandas as pd
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict, Any, Optional
import time
import warnings
from datetime import datetime
import logging
import json
from dataclasses import dataclass
from enum import Enum
import re

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SeverityLevel(Enum):
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ConfidenceLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class MedicalResponse:
    """Structured response from the medical system"""
    primary_condition: str
    confidence_level: ConfidenceLevel
    severity: SeverityLevel
    symptoms_match: float
    description: str
    recommendations: List[str]
    precautions: List[str]
    when_to_seek_care: str
    duration: str
    similar_conditions: List[str]
    sources: List[str]


# Enhanced medical dataset with more comprehensive information
enhanced_disease_data = [
    {
        'Disease': 'Common Cold',
        'Symptoms': 'sneezing, runny nose, cough, sore throat, fatigue, nasal congestion, mild headache',
        'Description': 'A viral infection of the upper respiratory tract caused by rhinoviruses, coronaviruses, or other respiratory viruses. It is highly contagious and typically resolves without treatment.',
        'Precautions': 'rest, increase fluid intake, avoid close contact with others, wash hands frequently, use tissue when sneezing or coughing, maintain good hygiene',
        'Duration': '7-10 days',
        'When_to_see_doctor': 'symptoms persist beyond 10 days, fever above 101.5F, severe headache, difficulty breathing, chest pain, persistent cough with blood',
        'ICD10': 'J00',
        'Differential_Diagnosis': 'Influenza, Allergic Rhinitis, Sinusitis, Strep Throat',
        'Risk_Factors': 'exposure to infected individuals, weakened immune system, stress, lack of sleep, poor nutrition',
        'Complications': 'secondary bacterial infections, sinusitis, otitis media, exacerbation of asthma'
    },
    {
        'Disease': 'Influenza',
        'Symptoms': 'high fever, body aches, chills, severe fatigue, headache, dry cough, sore throat, muscle pain',
        'Description': 'A highly contagious respiratory illness caused by influenza A or B viruses. More severe than common cold with rapid onset of symptoms.',
        'Precautions': 'annual influenza vaccination, rest, stay hydrated, avoid crowded places, antiviral medication if prescribed within 48 hours, isolation until fever-free for 24 hours',
        'Duration': '1-2 weeks',
        'When_to_see_doctor': 'difficulty breathing, chest pain, persistent high fever, severe weakness, symptoms worsen after initial improvement, signs of dehydration',
        'ICD10': 'J11',
        'Differential_Diagnosis': 'Common Cold, COVID-19, Pneumonia, Strep Throat',
        'Risk_Factors': 'age over 65, pregnancy, chronic conditions, immunocompromised state, obesity',
        'Complications': 'pneumonia, myocarditis, encephalitis, multi-organ failure'
    },
    {
        'Disease': 'Type 2 Diabetes',
        'Symptoms': 'increased thirst, frequent urination, fatigue, blurred vision, slow wound healing, unexplained weight loss, tingling in hands or feet',
        'Description': 'A chronic metabolic disorder characterized by insulin resistance and relative insulin deficiency, leading to elevated blood glucose levels.',
        'Precautions': 'monitor blood glucose regularly, maintain healthy diet low in refined sugars, exercise regularly, take prescribed medications as directed, regular medical follow-ups, foot care',
        'Duration': 'chronic lifelong condition',
        'When_to_see_doctor': 'blood glucose consistently above 180 mg/dL, symptoms of diabetic ketoacidosis, new vision changes, foot ulcers, medication side effects',
        'ICD10': 'E11',
        'Differential_Diagnosis': 'Type 1 Diabetes, MODY, Secondary Diabetes, Gestational Diabetes',
        'Risk_Factors': 'obesity, family history, sedentary lifestyle, age over 45, hypertension, PCOS',
        'Complications': 'diabetic nephropathy, diabetic retinopathy, peripheral neuropathy, cardiovascular disease'
    },
    {
        'Disease': 'Hypertension',
        'Symptoms': 'often asymptomatic, may include headache, shortness of breath, nosebleeds, chest pain, dizziness, vision changes',
        'Description': 'Persistently elevated blood pressure above 130/80 mmHg, often called the silent killer due to lack of symptoms.',
        'Precautions': 'reduce sodium intake to less than 2300mg daily, regular aerobic exercise, maintain healthy weight, limit alcohol consumption, take prescribed medications consistently, stress management',
        'Duration': 'chronic lifelong condition',
        'When_to_see_doctor': 'blood pressure readings consistently above 180/120, severe headaches, chest pain, vision problems, signs of stroke or heart attack',
        'ICD10': 'I10',
        'Differential_Diagnosis': 'Secondary Hypertension, White Coat Hypertension, Masked Hypertension',
        'Risk_Factors': 'family history, age, obesity, high sodium diet, physical inactivity, alcohol use, stress',
        'Complications': 'stroke, heart attack, kidney disease, heart failure, peripheral artery disease'
    },
    {
        'Disease': 'Pneumonia',
        'Symptoms': 'cough with purulent sputum, fever with chills, shortness of breath, chest pain that worsens with breathing, fatigue, confusion in elderly patients',
        'Description': 'Infection of the lung parenchyma causing inflammation of alveoli, which may fill with fluid or pus. Can be bacterial, viral, or fungal.',
        'Precautions': 'complete prescribed antibiotic course, rest, increase fluid intake, oxygen therapy if needed, pneumococcal and influenza vaccinations',
        'Duration': '1-3 weeks for recovery',
        'When_to_see_doctor': 'difficulty breathing, chest pain, persistent high fever, coughing up blood, worsening symptoms, signs of sepsis',
        'ICD10': 'J18',
        'Differential_Diagnosis': 'Bronchitis, Lung Cancer, Pulmonary Embolism, COPD exacerbation',
        'Risk_Factors': 'age over 65, smoking, chronic lung disease, immunocompromised state, recent viral illness',
        'Complications': 'sepsis, respiratory failure, lung abscess, pleural effusion'
    },
    {
        'Disease': 'Generalized Anxiety Disorder',
        'Symptoms': 'excessive worry, restlessness, fatigue, difficulty concentrating, irritability, muscle tension, sleep disturbances, panic attacks',
        'Description': 'Mental health condition characterized by persistent and excessive anxiety about various life events and activities.',
        'Precautions': 'cognitive behavioral therapy, relaxation techniques, regular exercise, limit caffeine and alcohol, maintain consistent sleep schedule, mindfulness practices',
        'Duration': 'chronic condition, varies with treatment',
        'When_to_see_doctor': 'symptoms interfere with daily functioning, panic attacks, thoughts of self-harm, substance abuse, inability to work or maintain relationships',
        'ICD10': 'F41.1',
        'Differential_Diagnosis': 'Panic Disorder, Social Anxiety Disorder, Depression, Hyperthyroidism',
        'Risk_Factors': 'family history, trauma, chronic stress, medical conditions, substance use',
        'Complications': 'depression, substance abuse, social isolation, physical health problems'
    },
    {
        'Disease': 'Major Depressive Disorder',
        'Symptoms': 'persistent sadness, loss of interest in activities, fatigue, changes in appetite, sleep disturbances, feelings of worthlessness, difficulty concentrating, thoughts of death',
        'Description': 'A mood disorder characterized by persistent depressive symptoms that significantly impact daily functioning.',
        'Precautions': 'psychotherapy, medication as prescribed, maintain social connections, regular physical activity, stress management, avoid alcohol and drugs',
        'Duration': 'episodes last weeks to months, can be recurrent',
        'When_to_see_doctor': 'symptoms persist for more than 2 weeks, thoughts of self-harm or suicide, inability to function, substance abuse, psychotic symptoms',
        'ICD10': 'F32',
        'Differential_Diagnosis': 'Bipolar Disorder, Dysthymia, Adjustment Disorder, Hypothyroidism',
        'Risk_Factors': 'family history, trauma, chronic stress, medical conditions, substance use, social isolation',
        'Complications': 'suicide, substance abuse, relationship problems, work impairment'
    },
    {
        'Disease': 'Migraine',
        'Symptoms': 'severe unilateral headache, nausea, vomiting, photophobia, phonophobia, visual aura, throbbing pain lasting 4-72 hours',
        'Description': 'A neurological disorder characterized by recurrent moderate to severe headaches often accompanied by autonomic symptoms.',
        'Precautions': 'identify and avoid triggers, maintain regular sleep schedule, stay hydrated, stress management, prescribed prophylactic medications, acute treatment medications',
        'Duration': '4-72 hours per episode',
        'When_to_see_doctor': 'sudden severe headache unlike previous ones, headache with fever and stiff neck, changes in headache pattern, neurological symptoms, medication overuse',
        'ICD10': 'G43',
        'Differential_Diagnosis': 'Tension Headache, Cluster Headache, Secondary Headache, Medication Overuse Headache',
        'Risk_Factors': 'family history, female gender, hormonal changes, certain foods, stress, sleep changes',
        'Complications': 'medication overuse headache, status migrainosus, stroke (rare)'
    }
]


class EnhancedMedicalKnowledgeBase:
    """Enhanced knowledge base with improved RAG capabilities"""

    def __init__(self):
        self.documents = []
        self.embeddings = None
        self.index = None
        self.model = None
        self.metadata_index = {}
        self.symptom_patterns = {}
        self.load_knowledge_base()

    def load_knowledge_base(self):
        """Load and process medical knowledge base"""
        try:
            # Load enhanced dataset
            df = pd.DataFrame(enhanced_disease_data)
            self.documents = self._process_documents(df)

            # Initialize advanced embedding model
            self.model = SentenceTransformer('all-MiniLM-L6-v2')

            # Build enhanced indexes
            self._build_vector_index()
            self._build_metadata_index()
            self._build_symptom_patterns()

            logger.info(f"Knowledge base loaded with {len(self.documents)} conditions")

        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            raise

    def _process_documents(self, df: pd.DataFrame) -> List[Dict]:
        """Process raw documents into structured format"""
        processed_docs = []

        for idx, row in df.iterrows():
            # Parse and clean data
            symptoms = [s.strip().lower() for s in row['Symptoms'].split(',')]
            precautions = [p.strip() for p in row['Precautions'].split(',')]
            risk_factors = [r.strip() for r in row.get('Risk_Factors', '').split(',')]
            complications = [c.strip() for c in row.get('Complications', '').split(',')]
            differential_dx = [d.strip() for d in row.get('Differential_Diagnosis', '').split(',')]

            # Calculate severity based on complications and duration
            severity = self._calculate_severity(row['Disease'], complications, row['Duration'])

            doc = {
                'id': idx + 1,
                'disease': row['Disease'],
                'symptoms': symptoms,
                'description': row['Description'],
                'precautions': precautions,
                'duration': row['Duration'],
                'when_to_see_doctor': row['When_to_see_doctor'],
                'icd10': row.get('ICD10', ''),
                'differential_diagnosis': differential_dx,
                'risk_factors': risk_factors,
                'complications': complications,
                'severity': severity,
                'category': self._categorize_disease(row['Disease']),
                'is_chronic': 'chronic' in row['Duration'].lower(),
                'is_contagious': self._is_contagious(row['Disease']),
                'urgency_level': self._assess_urgency(row['When_to_see_doctor'])
            }

            processed_docs.append(doc)

        return processed_docs

    def _calculate_severity(self, disease: str, complications: List[str], duration: str) -> SeverityLevel:
        """Calculate severity based on multiple factors"""
        high_severity_keywords = ['death', 'failure', 'sepsis', 'stroke', 'heart attack', 'cancer']
        critical_complications = ['respiratory failure', 'multi-organ failure', 'sepsis', 'stroke']

        if any(comp in ' '.join(complications).lower() for comp in critical_complications):
            return SeverityLevel.CRITICAL
        elif any(keyword in ' '.join(complications).lower() for keyword in high_severity_keywords):
            return SeverityLevel.HIGH
        elif 'chronic' in duration.lower():
            return SeverityLevel.MODERATE
        else:
            return SeverityLevel.LOW

    def _categorize_disease(self, disease_name: str) -> str:
        """Enhanced disease categorization"""
        disease_lower = disease_name.lower()

        categories = {
            'respiratory': ['cold', 'cough', 'pneumonia', 'asthma', 'influenza', 'bronchitis'],
            'cardiovascular': ['hypertension', 'heart', 'cardiac', 'blood pressure', 'stroke'],
            'endocrine': ['diabetes', 'thyroid', 'hormone', 'insulin'],
            'neurological': ['migraine', 'headache', 'seizure', 'stroke', 'vertigo'],
            'mental_health': ['anxiety', 'depression', 'mood', 'psychiatric'],
            'infectious': ['infection', 'viral', 'bacterial', 'fungal'],
            'autoimmune': ['arthritis', 'lupus', 'inflammatory']
        }

        for category, keywords in categories.items():
            if any(keyword in disease_lower for keyword in keywords):
                return category
        return 'general'

    def _is_contagious(self, disease: str) -> bool:
        """Determine if disease is contagious"""
        contagious_diseases = [
            'common cold', 'influenza', 'pneumonia', 'tuberculosis',
            'covid', 'strep', 'viral', 'bacterial infection'
        ]
        return any(contagious in disease.lower() for contagious in contagious_diseases)

    def _assess_urgency(self, when_to_see_doctor: str) -> str:
        """Assess urgency level based on when to see doctor criteria"""
        urgent_keywords = ['emergency', 'immediately', 'severe', 'difficulty breathing', 'chest pain']

        if any(keyword in when_to_see_doctor.lower() for keyword in urgent_keywords):
            return 'high'
        elif any(keyword in when_to_see_doctor.lower() for keyword in ['persist', 'worsen', 'continue']):
            return 'medium'
        else:
            return 'low'

    def _build_vector_index(self):
        """Build FAISS vector index for semantic search"""
        if not self.model:
            return

        # Create comprehensive text representations
        corpus = []
        for doc in self.documents:
            text_components = [
                doc['disease'],
                doc['description'],
                ' '.join(doc['symptoms']),
                ' '.join(doc['precautions']),
                doc['category'],
                ' '.join(doc['risk_factors']),
                ' '.join(doc['complications'])
            ]
            corpus.append(' '.join(text_components))

        # Generate embeddings
        self.embeddings = self.model.encode(corpus, show_progress_bar=True)

        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(self.embeddings.astype('float32'))

        logger.info(f"Vector index built with {len(corpus)} documents")

    def _build_metadata_index(self):
        """Build metadata index for filtering"""
        for doc in self.documents:
            # Index by category
            if doc['category'] not in self.metadata_index:
                self.metadata_index[doc['category']] = []
            self.metadata_index[doc['category']].append(doc['id'])

            # Index by severity
            severity_key = f"severity_{doc['severity'].value}"
            if severity_key not in self.metadata_index:
                self.metadata_index[severity_key] = []
            self.metadata_index[severity_key].append(doc['id'])

    def _build_symptom_patterns(self):
        """Build symptom pattern matching"""
        for doc in self.documents:
            for symptom in doc['symptoms']:
                if symptom not in self.symptom_patterns:
                    self.symptom_patterns[symptom] = []
                self.symptom_patterns[symptom].append(doc['id'])

    def search(self, query: str, top_k: int = 3, filters: Dict = None) -> List[Dict]:
        """Enhanced search with filtering and ranking"""
        if not self.model or not self.index:
            return []

        # Generate query embedding
        query_embedding = self.model.encode([query])

        # Perform vector search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k * 2)

        # Apply filters and ranking
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['semantic_score'] = float(score)

                # Apply filters
                if filters and not self._match_filters(doc, filters):
                    continue

                # Calculate symptom match score
                doc['symptom_match_score'] = self._calculate_symptom_match(query, doc['symptoms'])

                # Calculate combined score
                doc['combined_score'] = (
                        doc['semantic_score'] * 0.6 +
                        doc['symptom_match_score'] * 0.4
                )

                results.append(doc)

        # Sort by combined score and return top k
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        return results[:top_k]

    def _match_filters(self, doc: Dict, filters: Dict) -> bool:
        """Check if document matches filters"""
        for filter_key, filter_value in filters.items():
            if filter_key == 'category' and doc['category'] != filter_value:
                return False
            elif filter_key == 'severity' and doc['severity'] != filter_value:
                return False
            elif filter_key == 'is_chronic' and doc['is_chronic'] != filter_value:
                return False
        return True

    def _calculate_symptom_match(self, query: str, symptoms: List[str]) -> float:
        """Calculate symptom match score"""
        query_lower = query.lower()
        query_tokens = set(re.findall(r'\b\w+\b', query_lower))

        matches = 0
        for symptom in symptoms:
            symptom_tokens = set(re.findall(r'\b\w+\b', symptom.lower()))
            if query_tokens.intersection(symptom_tokens):
                matches += 1

        return matches / len(symptoms) if symptoms else 0.0


class EnhancedMedicalChatbot:
    """Enhanced chatbot with improved response generation"""

    def __init__(self, knowledge_base: EnhancedMedicalKnowledgeBase):
        self.kb = knowledge_base
        self.conversation_history = []
        self.context_window = 3  # Number of previous interactions to consider

    def generate_response(self, query: str, user_context: Dict = None) -> MedicalResponse:
        """Generate enhanced medical response"""

        # Analyze query intent and extract entities
        intent = self._analyze_intent(query)
        entities = self._extract_entities(query)

        # Apply context from conversation history
        enhanced_query = self._apply_context(query, self.conversation_history[-self.context_window:])

        # Search knowledge base
        search_results = self.kb.search(enhanced_query, top_k=3)

        if not search_results:
            return self._generate_fallback_response()

        # Generate structured response
        response = self._create_structured_response(search_results, intent, entities)

        # Update conversation history
        self.conversation_history.append({
            'query': query,
            'response': response,
            'timestamp': datetime.now()
        })

        return response

    def _analyze_intent(self, query: str) -> str:
        """Analyze user intent from query"""
        query_lower = query.lower()

        intent_patterns = {
            'symptoms': ['symptom', 'feel', 'experiencing', 'having', 'pain', 'ache', 'hurt'],
            'diagnosis': ['what is', 'what could', 'could this be', 'might be'],
            'treatment': ['treat', 'cure', 'medication', 'medicine', 'therapy'],
            'prevention': ['prevent', 'avoid', 'stop', 'reduce risk'],
            'prognosis': ['how long', 'duration', 'recovery', 'get better'],
            'urgency': ['emergency', 'urgent', 'serious', 'dangerous', 'worried']
        }

        for intent, patterns in intent_patterns.items():
            if any(pattern in query_lower for pattern in patterns):
                return intent

        return 'general'

    def _extract_entities(self, query: str) -> Dict:
        """Extract medical entities from query"""
        entities = {
            'symptoms': [],
            'body_parts': [],
            'duration': [],
            'severity': []
        }

        query_lower = query.lower()

        # Extract symptoms
        common_symptoms = [
            'fever', 'headache', 'cough', 'fatigue', 'nausea', 'vomiting',
            'dizziness', 'pain', 'shortness of breath', 'chest pain'
        ]

        for symptom in common_symptoms:
            if symptom in query_lower:
                entities['symptoms'].append(symptom)

        # Extract severity indicators
        severity_indicators = {
            'mild': ['mild', 'slight', 'little'],
            'moderate': ['moderate', 'medium'],
            'severe': ['severe', 'intense', 'unbearable', 'extreme']
        }

        for level, indicators in severity_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                entities['severity'].append(level)

        return entities

    def _apply_context(self, query: str, history: List[Dict]) -> str:
        """Apply conversation context to enhance query"""
        if not history:
            return query

        # Get relevant context from recent interactions
        context_terms = []
        for interaction in history:
            if interaction['response'].primary_condition:
                context_terms.append(interaction['response'].primary_condition)

        # Enhance query with context
        if context_terms:
            enhanced_query = f"{query} related to {' '.join(context_terms)}"
        else:
            enhanced_query = query

        return enhanced_query

    def _create_structured_response(self, search_results: List[Dict], intent: str, entities: Dict) -> MedicalResponse:
        """Create structured medical response"""
        primary_result = search_results[0]

        # Determine confidence level
        confidence = self._assess_confidence(primary_result, entities)

        # Create response
        response = MedicalResponse(
            primary_condition=primary_result['disease'],
            confidence_level=confidence,
            severity=primary_result['severity'],
            symptoms_match=primary_result['symptom_match_score'],
            description=primary_result['description'],
            recommendations=primary_result['precautions'][:5],
            precautions=primary_result['precautions'],
            when_to_seek_care=primary_result['when_to_see_doctor'],
            duration=primary_result['duration'],
            similar_conditions=[r['disease'] for r in search_results[1:3] if r['disease'] != primary_result['disease']],
            sources=[f"Medical Knowledge Base - {primary_result['icd10']}"]
        )

        return response

    def _assess_confidence(self, result: Dict, entities: Dict) -> ConfidenceLevel:
        """Assess confidence level of the response"""
        combined_score = result.get('combined_score', 0)
        symptom_match = result.get('symptom_match_score', 0)

        if combined_score > 0.8 and symptom_match > 0.6:
            return ConfidenceLevel.HIGH
        elif combined_score > 0.6 and symptom_match > 0.4:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    def _generate_fallback_response(self) -> MedicalResponse:
        """Generate fallback response when no matches found"""
        return MedicalResponse(
            primary_condition="Unknown",
            confidence_level=ConfidenceLevel.LOW,
            severity=SeverityLevel.LOW,
            symptoms_match=0.0,
            description="I couldn't find specific information about your symptoms.",
            recommendations=["Consult with a healthcare professional for proper evaluation"],
            precautions=["Seek medical attention if symptoms persist or worsen"],
            when_to_seek_care="Consider consulting a healthcare provider",
            duration="Unknown",
            similar_conditions=[],
            sources=[]
        )


class EnhancedResponseFormatter:
    """Enhanced response formatter with improved accuracy indicators"""

    @staticmethod
    def format_medical_response(response: MedicalResponse) -> str:
        """Format medical response with accuracy indicators"""

        # Header with confidence indicator
        confidence_indicator = {
            ConfidenceLevel.HIGH: "High confidence match",
            ConfidenceLevel.MEDIUM: "Moderate confidence match",
            ConfidenceLevel.LOW: "Low confidence match"
        }

        severity_indicator = {
            SeverityLevel.LOW: "Low severity",
            SeverityLevel.MODERATE: "Moderate severity",
            SeverityLevel.HIGH: "High severity",
            SeverityLevel.CRITICAL: "Critical severity"
        }

        formatted_response = f"""
# Medical Analysis: {response.primary_condition}

**Confidence Level:** {confidence_indicator[response.confidence_level]}
**Severity:** {severity_indicator[response.severity]}
**Symptom Match Score:** {response.symptoms_match:.2f}

## Condition Overview
{response.description}

## Primary Recommendations
"""

        for i, rec in enumerate(response.recommendations, 1):
            formatted_response += f"{i}. {rec}\n"

        formatted_response += f"""
## When to Seek Medical Care
{response.when_to_seek_care}

## Expected Duration
{response.duration}
"""

        if response.similar_conditions:
            formatted_response += "\n## Similar Conditions to Consider\n"
            for condition in response.similar_conditions:
                formatted_response += f"- {condition}\n"

        formatted_response += """
## Important Disclaimers
- This information is for educational purposes only
- Always consult healthcare professionals for medical advice
- Individual symptoms may vary significantly
- Seek immediate medical attention for severe symptoms
"""

        return formatted_response


class EnhancedMedicalChatbotApp:
    """Enhanced medical chatbot application"""

    def __init__(self):
        self.kb = EnhancedMedicalKnowledgeBase()
        self.chatbot = EnhancedMedicalChatbot(self.kb)
        self.formatter = EnhancedResponseFormatter()

    def run(self):
        st.set_page_config(
            page_title="Enhanced Medical Assistant",
            page_icon="🩺",
            layout="wide"
        )

        st.title("Enhanced Medical Assistant with Improved RAG")
        st.markdown("*Advanced medical information system with enhanced accuracy*")

        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Sidebar with system information
        with st.sidebar:
            st.header("System Information")

            # Knowledge base stats
            st.subheader("Knowledge Base")
            st.write(f"Conditions: {len(self.kb.documents)}")
            st.write(f"Categories: {len(set(doc['category'] for doc in self.kb.documents))}")

            # Accuracy features
            st.subheader("Accuracy Features")
            st.write("- Semantic similarity matching")
            st.write("- Symptom pattern recognition")
            st.write("- Confidence level assessment")
            st.write("- Multi-factor ranking")
            st.write("- Conversation context")

            # Clear chat
            if st.button("Clear Chat"):
                st.session_state.messages = []
                st.rerun()

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Describe your symptoms or ask a medical question"):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing your query..."):
                    response = self.chatbot.generate_response(prompt)
                    formatted_response = self.formatter.format_medical_response(response)
                    st.markdown(formatted_response)

            # Add assistant response
            st.session_state.messages.append({"role": "assistant", "content": formatted_response})


if __name__ == "__main__":
    app = EnhancedMedicalChatbotApp()
    app.run()