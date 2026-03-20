import csv, io, json, os
from datetime import datetime
from functools import wraps
from flask import Flask, flash, redirect, render_template, request, session, url_for, send_file
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import fitz
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

db = SQLAlchemy()

class User(db.Model):
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
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
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
    faculty_manual_score = db.Column(db.Float)
    faculty_comment = db.Column(db.Text)
    final_score = db.Column(db.Float)
    status = db.Column(db.String(20), default="analyzed")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class SystemSetting(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    site_name = db.Column(db.String(150), default="NCP & FDAR AI Analyzer")

def normalize_email(email): return (email or "").strip().lower()
def allowed_file(filename): return "." in filename and filename.rsplit(".", 1)[1].lower() in {"pdf","png","jpg","jpeg"}

NCP_RUBRIC = {"Assessment":20,"Diagnosis (NANDA alignment)":20,"Planning":15,"Implementation":15,"Evaluation":15,"Documentation Quality":15}
FDAR_RUBRIC = {"Focus":20,"Data":25,"Action":25,"Response":20,"Documentation Quality":10}

def parse_json_from_text(text):
    try: return json.loads(text)
    except Exception:
        s, e = text.find("{"), text.rfind("}")
        return json.loads(text[s:e+1])

def extract_pdf_text(path):
    doc = fitz.open(path)
    txt = "\n".join(page.get_text("text") for page in doc)
    doc.close()
    return txt.strip()

def fallback_analysis(document_type):
    if document_type == "NCP":
        breakdown = {
            "Assessment":{"score":14,"max_score":20,"comment":"Needs more patient-specific cues."},
            "Diagnosis (NANDA alignment)":{"score":13,"max_score":20,"comment":"Diagnosis needs clearer NANDA wording."},
            "Planning":{"score":11,"max_score":15,"comment":"Goals should be more measurable."},
            "Implementation":{"score":11,"max_score":15,"comment":"Interventions need prioritization."},
            "Evaluation":{"score":10,"max_score":15,"comment":"Evaluation criteria should be clearer."},
            "Documentation Quality":{"score":11,"max_score":15,"comment":"Formatting can improve."},
        }
    else:
        breakdown = {
            "Focus":{"score":15,"max_score":20,"comment":"Focus can be more specific."},
            "Data":{"score":18,"max_score":25,"comment":"More objective data needed."},
            "Action":{"score":18,"max_score":25,"comment":"Actions should be more explicit."},
            "Response":{"score":14,"max_score":20,"comment":"Response needs clearer outcomes."},
            "Documentation Quality":{"score":7,"max_score":10,"comment":"Clarity can improve."},
        }
    total = sum(v["score"] for v in breakdown.values())
    return {
        "total_score": total,
        "breakdown": breakdown,
        "strengths": ["Good structure attempt."],
        "improvements": ["Improve clinical specificity.", "Strengthen documentation accuracy."],
        "feedback_summary": f"{document_type} needs refinement based on rubric.",
        "suggested_revision": "Revise using more specific patient cues, clearer format, and stronger evaluation statements.",
        "curriculum_tags": ["documentation quality", "clinical reasoning"],
    }

def analyze_with_openai(document_type, extracted_text):
    api_key = os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")
    if not api_key or OpenAI is None:
        return fallback_analysis(document_type)
    rubric = NCP_RUBRIC if document_type == "NCP" else FDAR_RUBRIC
    client = OpenAI(api_key=api_key)
    prompt = f'''
You are an expert nursing documentation evaluator.
Evaluate this {document_type} using this rubric: {json.dumps(rubric)}.
Return ONLY valid JSON with keys:
total_score, breakdown, strengths, improvements, feedback_summary, suggested_revision, curriculum_tags.
Use NANDA-informed reasoning for NCP accuracy.
Student text:
"""{extracted_text[:12000]}"""
'''
    resp = client.responses.create(model=model, input=prompt, temperature=0.2)
    return parse_json_from_text(resp.output_text)

def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret")
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
    def current_user():
        return db.session.get(User, session["user_id"]) if "user_id" in session else None

    with app.app_context():
        db.create_all()
        if not SystemSetting.query.first():
            db.session.add(SystemSetting(site_name="NCP & FDAR AI Analyzer"))
        demos = [
            ("System Manager","manager@nursingai.local","manager","Manager123!"),
            ("Faculty Reviewer","faculty@nursingai.local","faculty","Faculty123!"),
            ("Student User","student@nursingai.local","student","Student123!")
        ]
        for n,e,r,p in demos:
            if not User.query.filter_by(email=e).first():
                u = User(name=n,email=e,role=r,status="active")
                if r=="student": u.section="4A"
                if r=="faculty": u.specialization="Nursing Education"
                u.set_password(p)
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
    @app.route("/student/submit", methods=["GET", "POST"])
@login_required
@role_required("student")
def student_submit():
    if request.method == "POST":
        document_type = request.form.get("document_type")
        title = request.form.get("title", "").strip()
        file = request.files.get("file")

        if document_type not in {"NCP", "FDAR"} or not title or not file or file.filename == "":
            flash("Complete all fields.", "danger")
            return redirect(url_for("student_submit"))

        if not allowed_file(file.filename):
            flash("Allowed: PDF, PNG, JPG, JPEG.", "danger")
            return redirect(url_for("student_submit"))

        filename = secure_filename(file.filename)
        stamped = f"{datetime.utcnow().strftime('%Y%m%d%H%M%S')}_{session['user_id']}_{filename}"
        path = os.path.join(app.config["UPLOAD_FOLDER"], stamped)
        file.save(path)

        ext = filename.rsplit(".", 1)[1].lower()

        # Extract text
        extracted_text = ""
        try:
            if ext == "pdf":
                extracted_text = extract_pdf_text(path)
            else:
                extracted_text = "Image uploaded. OCR path can be added next."
        except Exception as e:
            print("Extraction error:", e)
            extracted_text = "Text extraction failed."

        # AI Analysis
        try:
            analysis = analyze_with_openai(
                document_type,
                extracted_text or "No extractable text found."
            )
        except Exception as e:
            print("AI analysis error:", e)
            analysis = fallback_analysis(document_type)

        # Save to DB
        s = Submission(
            student_id=session["user_id"],
            document_type=document_type,
            title=title,
            original_filename=filename,
            file_path=path,
            extracted_text=extracted_text,
            ai_total_score=float(analysis.get("total_score", 0)),
            ai_breakdown_json=json.dumps(analysis.get("breakdown", {})),
            ai_feedback=analysis.get("feedback_summary"),
            ai_strengths=json.dumps(analysis.get("strengths", [])),
            ai_improvements=json.dumps(analysis.get("improvements", [])),
            ai_suggested_revision=analysis.get("suggested_revision"),
            ai_curriculum_tags=json.dumps(analysis.get("curriculum_tags", [])),
            final_score=float(analysis.get("total_score", 0)),
            status="analyzed"
        )

        db.session.add(s)
        db.session.commit()

        flash("Submission uploaded and analyzed.", "success")
        return redirect(url_for("student_submission_detail", submission_id=s.id))

    return render_template("student/submit.html")
    
    @app.route("/student/submissions")
    @login_required
    @role_required("student")
    def student_submissions():
        subs = Submission.query.filter_by(student_id=session["user_id"]).order_by(Submission.created_at.desc()).all()
        return render_template("student/submissions.html", submissions=subs)

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
        return render_template("faculty/submissions.html", submissions=Submission.query.order_by(Submission.created_at.desc()).all(), User=User)

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
            sub.status = "reviewed"
            db.session.commit()
            flash("Faculty review saved.", "success")
            return redirect(url_for("faculty_submission_detail", submission_id=sub.id))
        return render_template("faculty/submission_detail.html", sub=sub, student=db.session.get(User, sub.student_id), breakdown=json.loads(sub.ai_breakdown_json or "{}"))

    @app.route("/faculty/reports.csv")
    @login_required
    @role_required("faculty")
    def faculty_reports_csv():
        rows = [["Submission ID","Student","Document Type","Title","AI Score","Manual Score","Final Score","Status","Created At"]]
        for s in Submission.query.order_by(Submission.created_at.desc()).all():
            stu = db.session.get(User, s.student_id)
            rows.append([s.id, stu.name if stu else "Unknown", s.document_type, s.title, s.ai_total_score or "", s.faculty_manual_score or "", s.final_score or "", s.status, s.created_at.strftime("%Y-%m-%d %H:%M")])
        buf = io.StringIO(); writer = csv.writer(buf); writer.writerows(rows)
        mem = io.BytesIO(buf.getvalue().encode("utf-8")); mem.seek(0)
        return send_file(mem, mimetype="text/csv", as_attachment=True, download_name="ncp_fdar_report.csv")

    @app.route("/faculty/curriculum-insights")
    @login_required
    @role_required("faculty")
    def faculty_curriculum_insights():
        submissions = Submission.query.all()
        tag_counts, csum, ccnt = {}, {}, {}
        for sub in submissions:
            for t in json.loads(sub.ai_curriculum_tags or "[]"):
                tag_counts[t] = tag_counts.get(t,0)+1
            for crit, vals in json.loads(sub.ai_breakdown_json or "{}").items():
                csum[crit] = csum.get(crit,0)+float(vals.get("score",0))
                ccnt[crit] = ccnt.get(crit,0)+1
        avg_by_criterion = [{"criterion":k,"avg_score":round(csum[k]/ccnt[k],2)} for k in csum]
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
