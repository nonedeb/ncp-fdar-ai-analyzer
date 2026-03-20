import csv, io, json, os, re
from datetime import datetime
from functools import wraps
import fitz
from flask import Flask, flash, redirect, render_template, request, session, url_for, send_file
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
try:
    from openai import OpenAI
except Exception:
    OpenAI = None
try:
    import pytesseract
    from PIL import Image
except Exception:
    pytesseract = None
    Image = None

db = SQLAlchemy()

class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255))
    role = db.Column(db.String(20), nullable=False)
    section = db.Column(db.String(50))
    specialization = db.Column(db.String(150))
    status = db.Column(db.String(20), default="active")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    def set_password(self, password): self.password_hash = generate_password_hash(password)
    def check_password(self, password): return bool(self.password_hash) and check_password_hash(self.password_hash, password)

class Submission(db.Model):
    __tablename__ = "submissions"
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    document_type = db.Column(db.String(10), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(255), nullable=False)
    extracted_text = db.Column(db.Text)
    ai_total_score = db.Column(db.Float)
    ai_breakdown_json = db.Column(db.Text)
    ai_feedback = db.Column(db.Text)
    ai_strengths = db.Column(db.Text)
    ai_improvements = db.Column(db.Text)
    ai_suggested_revision = db.Column(db.Text)
    ai_curriculum_tags = db.Column(db.Text)
    ai_confidence = db.Column(db.Float)
    grade_band = db.Column(db.String(50))
    faculty_manual_score = db.Column(db.Float)
    faculty_comment = db.Column(db.Text)
    final_score = db.Column(db.Float)
    status = db.Column(db.String(20), default="analyzed")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class SystemSetting(db.Model):
    __tablename__ = "system_settings"
    id = db.Column(db.Integer, primary_key=True)
    site_name = db.Column(db.String(150), default="NCP & FDAR AI Analyzer")
    logo_url = db.Column(db.String(255))

NCP_RUBRIC = {"Assessment":20,"Diagnosis (NANDA alignment)":20,"Planning":15,"Implementation":15,"Evaluation":15,"Documentation Quality":15}
FDAR_RUBRIC = {"Focus":20,"Data":25,"Action":25,"Response":20,"Documentation Quality":10}
ALLOWED_EXTENSIONS = {"pdf","png","jpg","jpeg"}

def normalize_email(email): return (email or "").strip().lower()
def allowed_file(filename): return "." in filename and filename.rsplit(".",1)[1].lower() in ALLOWED_EXTENSIONS
def current_user(): return db.session.get(User, session["user_id"]) if "user_id" in session else None

def login_required(f):
    @wraps(f)
    def wrapper(*a, **kw):
        if "user_id" not in session:
            flash("Please log in first.", "warning")
            return redirect(url_for("login"))
        return f(*a, **kw)
    return wrapper

def role_required(*roles):
    def deco(f):
        @wraps(f)
        def wrapper(*a, **kw):
            if session.get("user_role") not in roles:
                flash("Unauthorized access.", "danger")
                return redirect(url_for("home"))
            return f(*a, **kw)
        return wrapper
    return deco

def extract_pdf_text(path):
    doc = fitz.open(path)
    text = "\n".join(page.get_text("text") for page in doc)
    doc.close()
    return text.strip()

def extract_image_text(path):
    if pytesseract is None or Image is None:
        return "OCR not configured on this deployment. Install Tesseract OCR to analyze image text."
    try:
        return pytesseract.image_to_string(Image.open(path)).strip()
    except Exception as e:
        return f"OCR failed: {e}"

def parse_json_from_text(text):
    try:
        return json.loads(text.strip())
    except Exception:
        s, e = text.find("{"), text.rfind("}")
        return json.loads(text[s:e+1])

def compute_confidence(text):
    if not text: return 25.0
    if "OCR not configured" in text or "failed" in text.lower(): return 35.0
    n = len(text.strip())
    if n < 200: return 45.0
    if n < 700: return 68.0
    if n < 1500: return 82.0
    return 91.0

def grade_band(score):
    score = float(score or 0)
    if score < 75: return "Needs Improvement"
    if score < 85: return "Fair"
    if score < 93: return "Good"
    return "Excellent"

def ncp_tags(text):
    low = (text or "").lower()
    tags = []
    if "related to" not in low or "as evidenced by" not in low: tags.append("NANDA PES structure")
    if not re.search(r"\b(goal|outcome|within \d+|after \d+)\b", low): tags.append("SMART outcomes")
    if "evaluate" not in low and "met" not in low: tags.append("Outcome evaluation")
    return tags

def fdar_tags(text):
    low = (text or "").lower()
    tags = []
    if "focus" not in low: tags.append("Focus statement writing")
    if "data" not in low: tags.append("Objective data gathering")
    if "action" not in low: tags.append("Intervention specificity")
    if "response" not in low: tags.append("Patient response documentation")
    return tags

def fallback_analysis(document_type, extracted_text):
    confidence = compute_confidence(extracted_text)
    if document_type == "NCP":
        breakdown = {
            "Assessment":{"score":14,"max_score":20,"comment":"Needs more patient-specific cues and clustered data."},
            "Diagnosis (NANDA alignment)":{"score":13,"max_score":20,"comment":"Diagnosis should better follow NANDA wording and PES structure."},
            "Planning":{"score":11,"max_score":15,"comment":"Goals should be more measurable and time-bound."},
            "Implementation":{"score":11,"max_score":15,"comment":"Interventions need clearer prioritization and clinical rationale."},
            "Evaluation":{"score":10,"max_score":15,"comment":"Evaluation criteria should reflect whether outcomes were met."},
            "Documentation Quality":{"score":11,"max_score":15,"comment":"Formatting and linkage between sections can improve."},
        }
        tags = list(dict.fromkeys(ncp_tags(extracted_text)+["NANDA alignment","Documentation quality"]))
        strengths = ["Shows an attempt to organize the care plan into appropriate sections.","Contains a recognizable nursing problem."]
        improvements = ["Use NANDA-consistent diagnosis structure with problem, etiology, and evidence.","Add more patient-specific assessment data and measurable outcomes.","Make evaluation statements clearly indicate whether goals were met."]
        suggested = "State clustered assessment cues, write one NANDA-based diagnosis using PES format, add SMART goals with timeframe, prioritize interventions, then evaluate whether goals were met."
    else:
        breakdown = {
            "Focus":{"score":15,"max_score":20,"comment":"Focus can be more specific and clinically prioritized."},
            "Data":{"score":18,"max_score":25,"comment":"Add more objective, patient-specific data."},
            "Action":{"score":18,"max_score":25,"comment":"Actions should clearly reflect what the nurse did."},
            "Response":{"score":14,"max_score":20,"comment":"Response should describe observable patient outcomes."},
            "Documentation Quality":{"score":7,"max_score":10,"comment":"Formatting and clarity can still improve."},
        }
        tags = list(dict.fromkeys(fdar_tags(extracted_text)+["FDAR response writing","Documentation clarity"]))
        strengths = ["Attempts to follow FDAR structure.","Includes nursing action statements."]
        improvements = ["Use a more precise focus statement.","Add both subjective and objective data.","Document a clearer patient response after the intervention."]
        suggested = "Make the focus specific, add subjective and objective data, document exact interventions, and state the patient's observable response."
    total = sum(v["score"] for v in breakdown.values())
    return {"total_score":total,"breakdown":breakdown,"strengths":strengths,"improvements":improvements,"feedback_summary":f"{document_type} needs refinement based on rubric and nursing documentation standards.","suggested_revision":suggested,"curriculum_tags":tags,"confidence":confidence}

def analyze_with_openai(document_type, extracted_text):
    api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
    if not api_key or OpenAI is None:
        return fallback_analysis(document_type, extracted_text)
    rubric = NCP_RUBRIC if document_type == "NCP" else FDAR_RUBRIC
    client = OpenAI(api_key=api_key)
    prompt = f"""
You are a senior clinical nursing instructor and evaluator.
Evaluate the following {document_type} using strict nursing standards and clinically sound educational judgment.
For NCP, use NANDA-informed reasoning and PES structure when appropriate.
For FDAR, evaluate whether Focus, Data, Action, and Response are clear and correct.
RUBRIC: {json.dumps(rubric)}
Return ONLY valid JSON with keys: total_score, breakdown, strengths, improvements, feedback_summary, suggested_revision, curriculum_tags, confidence.
Student submission:
\"\"\"{extracted_text[:15000]}\"\"\"
"""
    try:
        resp = client.responses.create(model=model, input=prompt, temperature=0.2)
        out = getattr(resp, "output_text", "")
        if not out:
            return fallback_analysis(document_type, extracted_text)
        data = parse_json_from_text(out)
        if "confidence" not in data:
            data["confidence"] = compute_confidence(extracted_text)
        return data
    except Exception as e:
        print("OpenAI error:", e)
        return fallback_analysis(document_type, extracted_text)

def build_report_csv_rows():
    rows = [["Submission ID","Student","Document Type","Title","AI Score","AI Confidence","Grade Band","Faculty Manual Score","Final Score","Status","Created At"]]
    for s in Submission.query.order_by(Submission.created_at.desc()).all():
        stu = db.session.get(User, s.student_id)
        rows.append([s.id, stu.name if stu else "Unknown", s.document_type, s.title, s.ai_total_score or "", s.ai_confidence or "", s.grade_band or "", s.faculty_manual_score or "", s.final_score or "", s.status, s.created_at.strftime("%Y-%m-%d %H:%M")])
    return rows

def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key")
    database_url = os.environ.get("DATABASE_URL", "sqlite:///app.db")
    if database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql+psycopg://", 1)
    elif database_url.startswith("postgresql://"):
        database_url = database_url.replace("postgresql://", "postgresql+psycopg://", 1)
    app.config["SQLALCHEMY_DATABASE_URI"] = database_url
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    db.init_app(app)

    with app.app_context():
        db.create_all()
        if not SystemSetting.query.first():
            db.session.add(SystemSetting(site_name="NCP & FDAR AI Analyzer"))
        demos = [("System Manager","manager@nursingai.local","manager","Manager123!"),("Faculty Reviewer","faculty@nursingai.local","faculty","Faculty123!"),("Student User","student@nursingai.local","student","Student123!")]
        for name, email, role, password in demos:
            if not User.query.filter_by(email=email).first():
                u = User(name=name, email=email, role=role, status="active")
                if role == "student": u.section = "4A"
                if role == "faculty": u.specialization = "Nursing Education"
                u.set_password(password)
                db.session.add(u)
        db.session.commit()

    @app.context_processor
    def inject_globals():
        return {"settings": SystemSetting.query.first(), "session_user_name": session.get("user_name"), "session_user_role": session.get("user_role")}

    @app.route("/")
    def home():
        role = session.get("user_role")
        if role == "student": return redirect(url_for("student_dashboard"))
        if role == "faculty": return redirect(url_for("faculty_dashboard"))
        if role == "manager": return redirect(url_for("manager_dashboard"))
        return redirect(url_for("login"))

    @app.route("/login", methods=["GET","POST"])
    def login():
        if request.method == "POST":
            user = User.query.filter_by(email=normalize_email(request.form.get("email")), status="active").first()
            if user and user.check_password(request.form.get("password","")):
                session["user_id"], session["user_name"], session["user_role"] = user.id, user.name, user.role
                flash("Login successful.", "success")
                return redirect(url_for("home"))
            flash("Invalid credentials.", "danger")
        return render_template("login.html")

    @app.route("/logout")
    def logout():
        session.clear()
        flash("Logged out.", "info")
        return redirect(url_for("login"))

    @app.route("/student")
    @login_required
    @role_required("student")
    def student_dashboard():
        subs = Submission.query.filter_by(student_id=session["user_id"]).order_by(Submission.created_at.desc()).all()
        avg = round(sum((s.final_score or s.ai_total_score or 0) for s in subs)/len(subs), 2) if subs else 0
        return render_template("student/dashboard.html", submissions=subs[:5], avg_score=avg)

    @app.route("/student/submit", methods=["GET","POST"])
    @login_required
    @role_required("student")
    def student_submit():
        if request.method == "POST":
            document_type = request.form.get("document_type")
            title = request.form.get("title","").strip()
            file = request.files.get("file")
            if document_type not in {"NCP","FDAR"} or not title or not file or file.filename == "":
                flash("Complete all fields.", "danger")
                return redirect(url_for("student_submit"))
            if not allowed_file(file.filename):
                flash("Allowed: PDF, PNG, JPG, JPEG.", "danger")
                return redirect(url_for("student_submit"))
            filename = secure_filename(file.filename)
            stamped = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{session['user_id']}_{filename}"
            path = os.path.join(app.config["UPLOAD_FOLDER"], stamped)
            file.save(path)
            ext = filename.rsplit(".",1)[1].lower()
            try:
                extracted_text = extract_pdf_text(path) if ext == "pdf" else extract_image_text(path)
            except Exception as e:
                print("Extraction error:", e)
                extracted_text = "Text extraction failed."
            try:
                analysis = analyze_with_openai(document_type, extracted_text or "No extractable text found.")
            except Exception as e:
                print("AI analysis error:", e)
                analysis = fallback_analysis(document_type, extracted_text)
            total_score = float(analysis.get("total_score",0))
            confidence = float(analysis.get("confidence", compute_confidence(extracted_text)))
            s = Submission(student_id=session["user_id"], document_type=document_type, title=title, original_filename=filename, file_path=path, extracted_text=extracted_text, ai_total_score=total_score, ai_breakdown_json=json.dumps(analysis.get("breakdown",{})), ai_feedback=analysis.get("feedback_summary"), ai_strengths=json.dumps(analysis.get("strengths",[])), ai_improvements=json.dumps(analysis.get("improvements",[])), ai_suggested_revision=analysis.get("suggested_revision"), ai_curriculum_tags=json.dumps(analysis.get("curriculum_tags",[])), ai_confidence=confidence, grade_band=grade_band(total_score), final_score=total_score, status="analyzed")
            db.session.add(s); db.session.commit()
            flash("Submission uploaded and analyzed.", "success")
            return redirect(url_for("student_submission_detail", submission_id=s.id))
        return render_template("student/submit.html")

    @app.route("/student/submissions")
    @login_required
    @role_required("student")
    def student_submissions():
        return render_template("student/submissions.html", submissions=Submission.query.filter_by(student_id=session["user_id"]).order_by(Submission.created_at.desc()).all())

    @app.route("/student/submissions/<int:submission_id>")
    @login_required
    @role_required("student")
    def student_submission_detail(submission_id):
        sub = db.session.get(Submission, submission_id)
        if not sub or sub.student_id != session["user_id"]:
            flash("Submission not found.", "danger")
            return redirect(url_for("student_submissions"))
        return render_template("student/submission_detail.html", sub=sub, breakdown=json.loads(sub.ai_breakdown_json or "{}"), strengths=json.loads(sub.ai_strengths or "[]"), improvements=json.loads(sub.ai_improvements or "[]"), tags=json.loads(sub.ai_curriculum_tags or "[]"))

    @app.route("/faculty")
    @login_required
    @role_required("faculty")
    def faculty_dashboard():
        total = Submission.query.count()
        reviewed = Submission.query.filter(Submission.status.in_(["reviewed","finalized"])).count()
        avg = db.session.query(db.func.avg(Submission.ai_total_score)).scalar() or 0
        return render_template("faculty/dashboard.html", total_submissions=total, reviewed=reviewed, avg_ai_score=round(avg,2))

    @app.route("/faculty/submissions")
    @login_required
    @role_required("faculty")
    def faculty_submissions():
        submissions = Submission.query.order_by(Submission.created_at.desc()).all()
        students = {u.id:u for u in User.query.filter_by(role="student").all()}
        doc_type = request.args.get("document_type")
        status = request.args.get("status")
        if doc_type: submissions = [s for s in submissions if s.document_type == doc_type]
        if status: submissions = [s for s in submissions if s.status == status]
        return render_template("faculty/submissions.html", submissions=submissions, students=students)

    @app.route("/faculty/submissions/<int:submission_id>", methods=["GET","POST"])
    @login_required
    @role_required("faculty")
    def faculty_submission_detail(submission_id):
        sub = db.session.get(Submission, submission_id)
        if not sub:
            flash("Submission not found.", "danger")
            return redirect(url_for("faculty_submissions"))
        if request.method == "POST":
            manual = request.form.get("faculty_manual_score","").strip()
            sub.faculty_manual_score = float(manual) if manual else None
            sub.faculty_comment = request.form.get("faculty_comment","").strip() or None
            sub.final_score = sub.faculty_manual_score if sub.faculty_manual_score is not None else sub.ai_total_score
            sub.grade_band = grade_band(sub.final_score)
            sub.status = "reviewed"
            db.session.commit()
            flash("Faculty review saved.", "success")
            return redirect(url_for("faculty_submission_detail", submission_id=sub.id))
        return render_template("faculty/submission_detail.html", sub=sub, student=db.session.get(User, sub.student_id), breakdown=json.loads(sub.ai_breakdown_json or "{}"), strengths=json.loads(sub.ai_strengths or "[]"), improvements=json.loads(sub.ai_improvements or "[]"), tags=json.loads(sub.ai_curriculum_tags or "[]"))

    @app.route("/faculty/reports.csv")
    @login_required
    @role_required("faculty")
    def faculty_reports_csv():
        rows = build_report_csv_rows()
        buf = io.StringIO(); writer = csv.writer(buf); writer.writerows(rows)
        mem = io.BytesIO(buf.getvalue().encode("utf-8")); mem.seek(0)
        return send_file(mem, mimetype="text/csv", as_attachment=True, download_name="ncp_fdar_report.csv")

    @app.route("/faculty/curriculum-insights")
    @login_required
    @role_required("faculty")
    def faculty_curriculum_insights():
        submissions = Submission.query.all()
        tag_counts, scores, counts = {}, {}, {}
        for sub in submissions:
            for t in json.loads(sub.ai_curriculum_tags or "[]"): tag_counts[t] = tag_counts.get(t,0)+1
            for crit, vals in json.loads(sub.ai_breakdown_json or "{}").items():
                scores[crit] = scores.get(crit,0)+float(vals.get("score",0))
                counts[crit] = counts.get(crit,0)+1
        avg_by_criterion = [{"criterion":k,"avg_score":round(scores[k]/max(counts[k],1),2)} for k in scores]
        avg_by_criterion.sort(key=lambda x: x["avg_score"])
        top_tags = sorted(tag_counts.items(), key=lambda x:x[1], reverse=True)[:10]
        narrative = "Students most frequently need reinforcement in: " + ", ".join([t[0] for t in top_tags]) if top_tags else "No curriculum trends yet."
        return render_template("faculty/curriculum_insights.html", avg_by_criterion=avg_by_criterion, top_tags=top_tags, narrative=narrative)

    @app.route("/manager")
    @login_required
    @role_required("manager")
    def manager_dashboard():
        return render_template("manager/dashboard.html", total_users=User.query.count(), total_students=User.query.filter_by(role="student").count(), total_faculty=User.query.filter_by(role="faculty").count(), total_submissions=Submission.query.count())

    @app.route("/manager/users")
    @login_required
    @role_required("manager")
    def manager_users():
        return render_template("manager/users.html", users=User.query.order_by(User.role.asc(), User.name.asc()).all())

    return app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
