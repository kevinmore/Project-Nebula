#include <GLWindow.h>
#include <UI/MainWindow.h>
int main(int argc, char* argv[])
{
	QApplication app(argc, argv);

	// setup resource and load the stylesheet
	QResource::registerResource("../Resource/StyleSheets/Dark/style.rcc");
	QFile file("../Resource/StyleSheets/Dark/style.qss");
	if(file.open(QIODevice::ReadOnly | QIODevice::Text))
	{
		app.setStyleSheet(file.readAll());
		file.close();
	}

	MainWindow window;
 	window.setWindowTitle("OpenGL Qt Framework - by Huanxiang Wang");
// 	window.resizeToScreenCenter();

	window.show();

	return app.exec();
}