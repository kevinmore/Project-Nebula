#include <GLWindow.h>
#include <UI/MainWindow.h>
int main(int argc, char* argv[])
{
	QApplication app(argc, argv);
	MainWindow window;
// 	window.setWindowTitle("OpenGL Qt Framework - by Huanxiang Wang");
// 	window.resizeToScreenCenter();

	window.show();

	return app.exec();
}