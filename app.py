from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import re
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import torch
from typing import List, Dict
import random
import datetime
from fuzzywuzzy import fuzz

app = Flask(__name__)
CORS(app)

class EnhancedMultilingualEidQABot:
    def __init__(self, data_file='dataSet.json'):
        print("🔄 Loading multilingual models...")
        self.bi_encoder = None
        self.cross_encoder = None
        print("📖 Processing dataset...")
        self.data = self._load_dataset(data_file)
        self.knowledge_chunks = self._create_chunks()
        self.chunk_embeddings = None
        self.question_patterns = self._initialize_question_patterns()
        print("✅ Bot ready!\n")
    def _ensure_embeddings(self):
        if self.chunk_embeddings is None:
            self._load_models()
            print("🧠 Creating embeddings...")
        self.chunk_embeddings = self.bi_encoder.encode(
            [chunk['text'] for chunk in self.knowledge_chunks],
            convert_to_tensor=True,
            show_progress_bar=True
        )

    
    def _load_dataset(self, data_file):
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return []
    
    def _create_chunks(self):
        chunks = []
        for item in self.data:
            text = item['text']
            tag = item.get('tag', 'General')
            chunks.append({
                'text': text,
                'tag': tag,
                'type': 'original',
                'score_boost': 1.0
            })
            if 'eid' in text.lower() or 'عید' in text:
                chunks.append({
                    'text': f"Eid information: {text}",
                    'tag': tag,
                    'type': 'enhanced',
                    'score_boost': 1.1
                })
            if 'prayer' in text.lower() or 'نماز' in text:
                chunks.append({
                    'text': f"Prayer information: {text}",
                    'tag': tag,
                    'type': 'enhanced',
                    'score_boost': 1.2
                })
            if 'qurbani' in text.lower() or 'قربانی' in text or 'sacrifice' in text.lower():
                chunks.append({
                    'text': f"Qurbani rules: {text}",
                    'tag': tag,
                    'type': 'enhanced',
                    'score_boost': 1.2
                })
            if 'funny' in tag.lower() or 'shair' in tag.lower():
                chunks.append({
                    'text': f"Fun fact: {text}",
                    'tag': tag,
                    'type': 'enhanced',
                    'score_boost': 0.9
                })
            if 'gaza' in text.lower() or 'غزہ' in text:
                chunks.append({
                    'text': f"Gaza context: {text}",
                    'tag': tag,
                    'type': 'enhanced',
                    'score_boost': 1.3
                })
        return chunks
    def _load_models(self):
        if self.bi_encoder is None:
            print("🔄 Loading bi-encoder model...")
            self.bi_encoder = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
        if self.cross_encoder is None:
            print("🔄 Loading cross-encoder model...")
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

    
    def _initialize_question_patterns(self):
        return {
            'greeting': ['eid mubarak', 'عید مبارک', 'hello', 'hi', 'salaam', 'سلام', 'mubarak', 'eid maz', 'eid mub', 'id mubarak'],
            'prayer': ['namaz', 'prayer', 'salah', 'eid ki namaz', 'نماز', 'how to pray', 'kaise parhein', 'nmaz', 'nmax', 'namaaz', 'salat'],
            'qurbani': ['qurbani', 'sacrifice', 'bakra', 'janwar', 'قربانی', 'ذبح', 'qurbni', 'kurbani', 'sacrifise'],
            'rules': ['rules', 'ahkam', 'قوانین', 'kya karna', 'what to do', 'kaise karna', 'rulez', 'ahkaam'],
            'time': ['time', 'waqt', 'kab', 'وقت', 'when', 'konsa din', 'kab hai'],
            'story': ['story', 'kahani', 'ibrahim', 'ismail', 'قصہ', 'واقعہ', 'history', 'kahaniya'],
            'food': ['food', 'khana', 'mithai', 'کھانا', 'سویاں', 'biryani', 'khane', 'meethi'],
            'funny': ['funny', 'shair', 'mazah', 'مزاح', 'joke', 'shairi', 'شاعری', 'mazak', 'maza'],
            'gaza': ['gaza', 'palestine', 'غزہ', 'فلسطین', 'war zone', 'gazah'],
            'general': ['kya hai', 'what is', 'بتائیں', 'معلومات', 'eid kya', 'عید کیا', 'eid hai']
        }
    
    def _clean_input(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text.strip().lower())
        text = re.sub(r'[^\w\s؟!]', '', text)  # Keep Urdu/English chars, spaces, and basic punctuation
        return text
    
    def _fuzzy_match(self, word: str, keywords: List[str]) -> bool:
        return any(fuzz.ratio(word, keyword) > 80 for keyword in keywords)
    
    def _detect_question_type(self, question: str) -> str:
        cleaned_question = self._clean_input(question)
        words = cleaned_question.split()
        for category, keywords in self.question_patterns.items():
            if any(self._fuzzy_match(word, keywords) for word in words):
                return category
        return 'general'
    
    def _get_contextual_boost(self, chunk: Dict, question_type: str) -> float:
        boost = chunk.get('score_boost', 1.0)
        if question_type == 'greeting' and 'greeting' in chunk['tag'].lower():
            boost *= 1.4
        elif question_type == 'prayer' and 'prayer' in chunk['tag'].lower():
            boost *= 1.3
        elif question_type == 'qurbani' and ('qurbani' in chunk['tag'].lower() or 'sacrifice' in chunk['tag'].lower()):
            boost *= 1.3
        elif question_type == 'story' and 'story' in chunk['tag'].lower():
            boost *= 1.2
        elif question_type == 'funny' and 'funny' in chunk['tag'].lower():
            boost *= 1.1
        elif question_type == 'gaza' and 'gaza' in chunk['tag'].lower():
            boost *= 1.3
        return boost
    
    def _is_time_sensitive(self, question: str) -> bool:
        time_keywords = ['time', 'waqt', 'kab', 'وقت', 'when', 'konsa din', 'kab hai']
        return any(self._fuzzy_match(word, time_keywords) for word in question.lower().split())
    
    def answer_question(self, question: str) -> str:
        self._load_models()
        self._ensure_embeddings()

        cleaned_question = self._clean_input(question)
        if not cleaned_question:
            return self._get_default_response('empty')
        
        question_type = self._detect_question_type(cleaned_question)
        question_embedding = self.bi_encoder.encode(cleaned_question, convert_to_tensor=True)
        cos_scores = util.cos_sim(question_embedding, self.chunk_embeddings)[0]
        
        boosted_scores = []
        for i, score in enumerate(cos_scores):
            boost = self._get_contextual_boost(self.knowledge_chunks[i], question_type)
            boosted_scores.append(score * boost)
        
        boosted_scores = torch.tensor(boosted_scores)
        top_k = min(15, len(self.knowledge_chunks))
        top_results = torch.topk(boosted_scores, k=top_k)
        top_indices = top_results.indices.tolist()
        top_chunks = [self.knowledge_chunks[i]['text'] for i in top_indices]
        top_scores = top_results.values.tolist()
        
        rerank_pairs = [(cleaned_question, chunk) for chunk in top_chunks]
        rerank_scores = self.cross_encoder.predict(rerank_pairs)
        
        combined_scores = []
        for i, rerank_score in enumerate(rerank_scores):
            combined_score = (rerank_score * 0.7) + (top_scores[i] * 0.3)
            combined_scores.append(combined_score)
        
        best_idx = max(range(len(combined_scores)), key=lambda i: combined_scores[i])
        best_chunk = top_chunks[best_idx]
        best_score = combined_scores[best_idx]
        
        avg_score = sum(combined_scores) / len(combined_scores)
        threshold = avg_score * 0.8
        
        if best_score < threshold:
            return self._get_default_response(question_type)
        
        # Clean the response - remove prefixes like "Eid information:", "Prayer information:", etc.
        response = best_chunk
        prefixes_to_remove = [
            "Eid information: ",
            "Prayer information: ", 
            "Qurbani rules: ",
            "Fun fact: ",
            "Gaza context: "
        ]
        
        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix):]
                break
        
        if self._is_time_sensitive(cleaned_question):
            current_date = datetime.datetime.now()
            islamic_date = "10th Dhul-Hijjah"  # Placeholder
            response += f"\n\n🕒 آج {current_date.strftime('%B %d, %Y')} ہے۔ عید الاضحیٰ عام طور پر {islamic_date} کو ہوتی ہے۔"
        
        response += "\n\n This is a demo. I'm working on this project, and its continuation depends on user feedback. Please share your suggestions by visiting our 'Contact Us' screen."
        return response
    
    def _get_default_response(self, question_type: str) -> str:
        defaults = {
            'greeting': "🌙Eid Mubarak! May Allah accept your prayers.",
            'prayer': "🕌   Eid prayer is 2 rakahs with extra takbeerat. Consult scholars for details.",
            'qurbani': "🐐  Qurbani is obligatory for those who meet nisab. The animal must be healthy.",
            'rules': "📜 Qurbani rules: Animal age, health, and intention are key.",
            'time': "⏰ Eid ul-Adha is from 10th to 12th Dhul-Hijjah.",
            'story': "📖 Eid ul-Adha commemorates Prophet Ibrahim's (AS) sacrifice.",
            'food': "🍲  Eid foods include sheer khurma, biryani, and sweets.",
            'funny': "😄  Eid fun: Eat sweets, collect Eidi!",
            'gaza': "🤲  Pray for the people of Gaza. They are in hardship.",
            'empty': " Ask something about Eid!",
            'general': "🌟I am your Eid Assistant, created by OCi Lab .  I am currently in progress and have limited data, focusing on small fun activities for Eid. I will improve myself after Eid"
        }
        return defaults.get(question_type, defaults['general'])
    
    def get_random_eid_fact(self) -> str:
        facts = [chunk for chunk in self.knowledge_chunks if chunk['tag'] in ['Eid_Overview', 'Prophet_Story', 'Eid_Prayer', 'Qurbani_Rules']]
        if facts:
            fact_text = random.choice(facts)['text']
            # Clean prefixes from random facts too
            prefixes_to_remove = [
                "Eid information: ",
                "Prayer information: ", 
                "Qurbani rules: ",
                "Fun fact: ",
                "Gaza context: "
            ]
            for prefix in prefixes_to_remove:
                if fact_text.startswith(prefix):
                    fact_text = fact_text[len(prefix):]
                    break
            return f"💡 {fact_text}"
        return "🌙 Eid Mubarak!"
    
    def get_random_greeting(self) -> str:
        greetings = [chunk for chunk in self.knowledge_chunks if 'greeting' in chunk['tag'].lower()]
        if greetings:
            greeting_text = random.choice(greetings)['text']
            # Clean prefixes from greetings too
            prefixes_to_remove = [
                "Eid information: ",
                "Prayer information: ", 
                "Qurbani rules: ",
                "Fun fact: ",
                "Gaza context: "
            ]
            for prefix in prefixes_to_remove:
                if greeting_text.startswith(prefix):
                    greeting_text = greeting_text[len(prefix):]
                    break
            return f"🎉 {greeting_text}"
        return "🌙 Eid Mubarak!"
    
    def get_random_shair(self) -> str:
        shairs = [chunk for chunk in self.knowledge_chunks if 'funny_shair_o_shairi' in chunk['tag'].lower()]
        if shairs:
            shair_text = random.choice(shairs)['text']
            # Clean prefixes from shairs too
            prefixes_to_remove = [
                "Eid information: ",
                "Prayer information: ", 
                "Qurbani rules: ",
                "Fun fact: ",
                "Gaza context: "
            ]
            for prefix in prefixes_to_remove:
                if shair_text.startswith(prefix):
                    shair_text = shair_text[len(prefix):]
                    break
            return f"😄 شاعری: {shair_text}"
        return "😂 No shairi found, just Eid Mubarak!"
    
    def get_contextual_info(self) -> str:
        current_date = datetime.datetime.now()
        islamic_date = "10th Dhul-Hijjah"  # Placeholder
        return f"🕒 {current_date.strftime('%B %d, %Y')}۔{islamic_date} "

# Instantiate the bot
bot = EnhancedMultilingualEidQABot('dataSet.json')

# Flask Routes
@app.route('/ask', methods=['POST'])
def ask_question():
    try:
        data = request.get_json()
        question = data.get('question', '')
        if not question:
            return jsonify({'answer': bot._get_default_response('empty')})
        answer = bot.answer_question(question)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e), 'answer': 'Sorry, something went wrong!'})

@app.route('/random', methods=['GET'])
def random_fact():
    fact = bot.get_random_eid_fact()
    return jsonify({'answer': fact})

@app.route('/greet', methods=['GET'])
def random_greeting():
    greeting = bot.get_random_greeting()
    return jsonify({'answer': greeting})

@app.route('/shair', methods=['GET'])
def random_shair():
    shair = bot.get_random_shair()
    return jsonify({'answer': shair})

@app.route('/context', methods=['GET'])
def contextual_info():
    info = bot.get_contextual_info()
    return jsonify({'answer': info})
@app.route('/warmup', methods=['GET'])
def warmup():
    try:
        bot._load_models()
        bot._ensure_embeddings()
        return jsonify({'status': 'Models warmed up and embeddings ready.'})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
