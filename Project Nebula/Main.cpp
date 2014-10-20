#include <GLWindow.h>
#include <Qt/qapplication.h>

int main(int argc, char* argv[])
{
	QApplication app(argc, argv);
	GLWindow window;
	window.setWindowTitle("OpenGL Qt Framework - by Huanxiang Wang");
	window.resizeToScreenCenter();

	window.show();

	return app.exec();
}