#include <UI/MainWindow.h>
int main(int argc, char* argv[])
{
	Magick::InitializeMagick(*argv);
	QApplication app(argc, argv);

	// setup resource and load the stylesheet
	QResource::registerResource("../Resource/StyleSheets/Dark/style.rcc");
	QResource::registerResource("../Resource/StyleSheets/Icons/icons.rcc");
	QFile file("../Resource/StyleSheets/Dark/style.qss");
	if(file.open(QIODevice::ReadOnly | QIODevice::Text))
	{
		app.setStyleSheet(file.readAll());
		file.close();
	}

	app.setWindowIcon(QIcon("../Resource/Logo/nebula.png"));
	
	MainWindow window;
	window.show();
    //window.showMaximized();

	return app.exec();
}

