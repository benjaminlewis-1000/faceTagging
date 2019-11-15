from bottle import route, run, default_app, request, jinja2_view, static_file, BaseRequest
from os import listdir, stat, path
from datetime import datetime
import magic


@route('/uploads/<file>', name='static')
def serve_upload(file):
    return static_file(file, root='uploads')


# Index
@route('/', method=['GET', 'POST'])
@jinja2_view('files.html', template_lookup=['views'])
def file_list():
    message = None
    if request.method == 'POST':
        upload = request.files.get('my_upload')
        if not path.exists('{0}/{1}'.format(uploads_dir, upload.filename)):
            upload.save(uploads_dir)
            message = ('Saved ok' if path.exists('{0}/{1}'.format(uploads_dir, upload.filename)) else 'Error Saving')
        else:
            message = "Error file already exists"

    files = list()
    for f in listdir(uploads_dir):
        full = path.join(uploads_dir, f)
        files.append({'name': f,
                      'size': stat(full).st_size,
                      'type': mime.from_file(full).decode('utf-8'),
                      'path': 'uploads/'+f,
                      'date': datetime.fromtimestamp(path.getctime(full))})
    return {'files': files, 'message': message}

# Create app instance
app = application = default_app()
BaseRequest.MEMFILE_MAX = 8096 * 1024  # 8mb
uploads_dir = 'uploads'
mime = magic.Magic(mime=True)
# Run bottle internal test server when invoked directly ie: non-uxsgi mode
if __name__ == '__main__':
    run(app=app, host='0.0.0.0', port=8081)