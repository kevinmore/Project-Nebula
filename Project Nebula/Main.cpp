#include <UI/MainWindow.h>
#include <statemachineviewer.h>

StateMachineViewer* showStateMachine(QStateMachine* machine)
{
	StateMachineViewer* smv = new StateMachineViewer();
	smv->setStateMachine(machine);
	smv->show();
	return smv;
}

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
    window.show();


	QPushButton button;

	
	QStateMachine machine;
	QState* s1 = new QState(&machine);
	s1->assignProperty(&button, "text", "State 1");
	
	QState* s2 = new QState(&machine);
	s2->assignProperty(&button, "text", "State 2");

	QState* s3 = new QState(&machine);
	s3->assignProperty(&button, "text", "State 3");

	s1->addTransition(&button, SIGNAL(clicked()), s2);
	s2->addTransition(&button, SIGNAL(clicked()), s3);
	s3->addTransition(&button, SIGNAL(clicked()), s1);

	QObject::connect(s1, SIGNAL(entered()), &button, SLOT(show()));
 	
// 	QObject::connect(final, SIGNAL(entered()), &app, SLOT(quit()));


	machine.setInitialState(s1);
	machine.start();

	StateMachineViewer* smv = showStateMachine(&machine);

	
	if(machine.configuration().contains(s1))
	{
		//do something
		qDebug() << "In state 2";
	}

	return app.exec();
}