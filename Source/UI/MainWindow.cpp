#include "MainWindow.h"
#include "HierarchyWidget.h"
#include "SkyboxDialog.h"
#include <Utility/LogCenter.h>

MainWindow::MainWindow(QWidget *parent)
	: QMainWindow(parent),
	  m_canvas(nullptr),
	  m_scene(nullptr),
	  m_camera(nullptr)
{
	// redirect the system out put message to the status bar
	connect(LogCenter::instance(), SIGNAL(message(QtMsgType, QMessageLogContext, QString)), 
		this, SLOT(showMessage(QtMsgType, QMessageLogContext, QString)));

	initializeCanvas();
	initializeRightDockableArea();
	initializeMenuBar();
	initializeToolBar();

	resize(1600, 900);

	// this is a trick to refresh the opengl surface to make it fit the window properly!
	showFullScreen();
	showNormal();
}

MainWindow::~MainWindow() 
{
}

void MainWindow::initializeCanvas() 
{
	m_canvas.reset(new Canvas);
	m_scene    = m_canvas->getScene();
	m_camera   = m_scene->getCamera();


	QWidget* canvas = QWidget::createWindowContainer(m_canvas.data());
	QDockWidget* m_dockCanvas = new QDockWidget("Scene", this);
	m_canvas->setContainerWidget(m_dockCanvas);
	m_dockCanvas->setWidget(canvas);
	m_dockCanvas->setFeatures(QDockWidget::DockWidgetMovable | QDockWidget::DockWidgetFloatable);
	addDockWidget(Qt::LeftDockWidgetArea, m_dockCanvas);
	setCentralWidget(m_dockCanvas);
}


void MainWindow::initializeToolBar()
{
	QToolBar* toolBar = new QToolBar("Nebula", this);

	QAction* playAction = toolBar->addAction(QIcon("../Resource/StyleSheets/Icons/play.png"), "Play", m_scene, SLOT(play()));
	QAction* pauseAction = toolBar->addAction(QIcon("../Resource/StyleSheets/Icons/pause.png"), "Pause", m_scene, SLOT(pause()));
	QAction* stepAction = toolBar->addAction(QIcon("../Resource/StyleSheets/Icons/forwardtoend.png"), "Step", m_scene, SLOT(step()));
	
	toolBar->addSeparator();

	QAction* loadModelAction = toolBar->addAction(QIcon("../Resource/StyleSheets/Icons/model.png"), "Load Model", m_scene, SLOT(showLoadModelDialog()));

	toolBar->addSeparator();

	QAction* openSceneAction = toolBar->addAction(QIcon("../Resource/StyleSheets/Icons/folder.png"), "Open Scene", m_scene, SLOT(showOpenSceneDialog()));
	QAction* saveSceneAction = toolBar->addAction(QIcon("../Resource/StyleSheets/Icons/disk.png"), "Save Scene", m_scene, SLOT(showSaveSceneDialog()));

	toolBar->addSeparator();

	QAction* resetCameraAction = toolBar->addAction(QIcon("../Resource/StyleSheets/Icons/camera.png"), "Reset Camera", m_camera, SLOT(resetCamera()));
	QAction* reloadSceneAction = toolBar->addAction(QIcon("../Resource/StyleSheets/Icons/reload.png"), "Reload Scene", m_scene, SLOT(reloadScene()));

	toolBar->addSeparator();

	QAction* debugSceneAction = toolBar->addAction(QIcon("../Resource/StyleSheets/Icons/magnifyingglass.png"), "Toggle Debug Mode", m_scene, SLOT(toggleDebugMode(bool)));
	debugSceneAction->setCheckable(true);

	
	addToolBar(Qt::LeftToolBarArea, toolBar);
}


void MainWindow::initializeMenuBar()
{
	// ############ File Menu ############
	QMenu *fileMenu = menuBar()->addMenu("&File");

	QAction *openSceneAction = new QAction("&Open Scene", this);
	fileMenu->addAction(openSceneAction);

	QAction *saveAction = new QAction("&Save Scene", this);
	fileMenu->addAction(saveAction);

	fileMenu->addSeparator();

	QAction *loadModelAction = new QAction("&Load Model", this);
	fileMenu->addAction(loadModelAction);

	fileMenu->addSeparator();

	QAction *exitAction = new QAction("&Exit", this);
	exitAction->setShortcut(QKeySequence("Ctrl+Q"));
	fileMenu->addAction(exitAction);

	// ############ Game Object ############
	QMenu *gameObjectMenu = menuBar()->addMenu("&Game Object");
	QAction *createEmpty = new QAction("Create Empty", this);
	gameObjectMenu->addAction(createEmpty);

	QMenu* createGameObjectMenu = gameObjectMenu->addMenu("Create Other");
	QAction *createParticleSystemAction = new QAction("Particle System", this);
	createGameObjectMenu->addAction(createParticleSystemAction);

	QAction *createLightAction = new QAction("Light", this);
	createGameObjectMenu->addAction(createLightAction);

	// ############ Scene Menu ############
	QMenu *sceneMenu = menuBar()->addMenu("&Scene");
	
	QAction* clearSceneAction = new QAction("&Clear All Objects", this);
	sceneMenu->addAction(clearSceneAction);

	QAction* resetSceneAction = new QAction("&Reset to Default", this);
	sceneMenu->addAction(resetSceneAction);

	// ############ Window Menu ############
	QMenu *windowMenu = menuBar()->addMenu("&Window");

	QAction *fullscreenAction = new QAction("&Fullscreen", this);
	fullscreenAction->setCheckable(true);
	fullscreenAction->setShortcut(QKeySequence(Qt::Key_F11));
	windowMenu->addAction(fullscreenAction);

	QAction* toggleHierarchyInspector = m_heirarchyViewer->toggleViewAction();
	toggleHierarchyInspector->setText("Show Hierarchy Inspector");
	windowMenu->addAction(toggleHierarchyInspector);

	QAction* toggleSettingsTab = m_dockSettingsArea->toggleViewAction();
	toggleSettingsTab->setText("Show Settings Window");
	windowMenu->addAction(toggleSettingsTab);


	// ############ Preference Menu ############
	QMenu *prefMenu = menuBar()->addMenu("&Preference");
	QMenu *bgMenu = prefMenu->addMenu("Background");

	QAction *bgColorAction = new QAction("Back Ground Color", this);
	bgMenu->addAction(bgColorAction);

	QAction *toggleSkyboxAction = new QAction("Enable Sky Box", this);
	toggleSkyboxAction->setCheckable(true);
	toggleSkyboxAction->setChecked(false);
	bgMenu->addAction(toggleSkyboxAction);

	QAction *skyboxSettingsAction = new QAction("Sky Box Settings", this);
	bgMenu->addAction(skyboxSettingsAction);

	// ############ System Menu ############
	QMenu *sytemMenu = menuBar()->addMenu("&System");
	
	QMenu *optionMenu = sytemMenu->addMenu("&Options");
	QAction *msaaAction = new QAction("&MSAA x4", this);
	msaaAction->setCheckable(true);
	msaaAction->setChecked(true);
	optionMenu->addAction(msaaAction);

	// model processing group
	QMenu *modelProcessMenu = optionMenu->addMenu("&Model Processing Quality");
	QActionGroup* modelProcessOptions = new QActionGroup(this);
	modelProcessOptions->setExclusive(true);
	QAction* simpleProcess = new QAction("Simple", this);
	simpleProcess->setCheckable(true);
	QAction* fastProcess = new QAction("Fast", this);
	fastProcess->setCheckable(true);
	QAction* qualityProcess = new QAction("Quality", this);
	qualityProcess->setCheckable(true);
	qualityProcess->setChecked(true);
	QAction* maxQualityProcess = new QAction("Max Quality", this);
	maxQualityProcess->setCheckable(true);
	modelProcessMenu->addAction(simpleProcess);
	modelProcessMenu->addAction(fastProcess);
	modelProcessMenu->addAction(qualityProcess);
	modelProcessMenu->addAction(maxQualityProcess);
	modelProcessOptions->addAction(simpleProcess);
	modelProcessOptions->addAction(fastProcess);
	modelProcessOptions->addAction(qualityProcess);
	modelProcessOptions->addAction(maxQualityProcess);
	connect(modelProcessOptions, SIGNAL(triggered(QAction*)), this, SLOT(actionTriggered(QAction*)));

	QAction *writeLogAction = new QAction("&Write Log Into File", this);
	writeLogAction->setCheckable(true);
	writeLogAction->setChecked(false);
	optionMenu->addAction(writeLogAction);

	QAction *debugModeAction = new QAction("&Debug Mode", this);
	debugModeAction->setCheckable(true);
	debugModeAction->setChecked(false);
	sytemMenu->addAction(debugModeAction);

	QAction *gpuInfoAction = new QAction("&GPU Info", this);
	sytemMenu->addAction(gpuInfoAction);

	QAction *systemLogAction = new QAction("&System Log", this);
	sytemMenu->addAction(systemLogAction);

	// ############ Signals & Slots ############
	connect(openSceneAction,  SIGNAL(triggered()),     m_scene, SLOT(showOpenSceneDialog()));
	connect(loadModelAction,  SIGNAL(triggered()),     m_scene, SLOT(showLoadModelDialog()));
	connect(saveAction,       SIGNAL(triggered()),     m_scene, SLOT(showSaveSceneDialog()));
	connect(resetSceneAction, SIGNAL(triggered()),     m_scene, SLOT(resetToDefaultScene()));
	connect(clearSceneAction, SIGNAL(triggered()),     m_scene, SLOT(clearScene()));
	connect(createEmpty,      SIGNAL(triggered()),     m_scene, SLOT(createEmptyGameObject()));
	connect(createParticleSystemAction, SIGNAL(triggered()), m_scene, SLOT(createParticleSystem()));
	connect(createLightAction, SIGNAL(triggered()), m_scene, SLOT(createLight()));
	connect(exitAction,       SIGNAL(triggered()),     qApp,    SLOT(quit()));
	connect(fullscreenAction, SIGNAL(triggered(bool)), this,    SLOT(setFullScreen(bool)));
	connect(msaaAction,       SIGNAL(triggered(bool)), m_scene, SLOT(toggleAA(bool)));
	connect(systemLogAction,  SIGNAL(triggered()),     this,    SLOT(showSystemLog()));
	connect(writeLogAction,   SIGNAL(triggered(bool)), LogCenter::instance(), SLOT(toggleWriteToFile(bool)));
	connect(gpuInfoAction,    SIGNAL(triggered()),     m_canvas.data(), SLOT(showGPUInfo()));
	connect(bgColorAction,    SIGNAL(triggered()),     this, SLOT(showBackGroundColorPicker()));
	connect(skyboxSettingsAction,    SIGNAL(triggered()),     this, SLOT(showSkyboxDialog()));
	connect(toggleSkyboxAction, SIGNAL(triggered(bool)), m_scene, SLOT(toggleSkybox(bool)));
	connect(debugModeAction, SIGNAL(triggered(bool)), m_scene, SLOT(toggleDebugMode(bool)));
}

void MainWindow::initializeRightDockableArea()
{
	initializeParamsArea();

	// create hierarchy inspector
	m_heirarchyViewer = new QDockWidget("Hierarchy Inspector", this);
	HierarchyWidget* inspector = new HierarchyWidget(m_canvas.data(), this);
	m_heirarchyViewer->setWidget(inspector);
	m_heirarchyViewer->setFeatures(QDockWidget::AllDockWidgetFeatures);
	//addDockWidget(Qt::RightDockWidgetArea, m_heirarchyViewer);
	tabifyDockWidget(m_dockSettingsArea, m_heirarchyViewer);
}

void MainWindow::initializeParamsArea()
{
	
	// ############ SETTINGS AREA ############

	m_dockSettingsArea = new QDockWidget("Settings", this);
	addDockWidget(Qt::RightDockWidgetArea, m_dockSettingsArea);

	m_params = new QWidget;
	m_dockSettingsArea->setWidget(m_params);

	fpsCounter = new QLCDNumber(2);
	fpsCounter->setSegmentStyle(QLCDNumber::Flat);
	fpsCounter->setFrameStyle(QFrame::Box | QFrame::Sunken);
	fpsCounter->display(0.0);

	QTabWidget* tab = new QTabWidget;

	QWidget* lightTab   = new QWidget;
	QWidget* optionsTab = new QWidget;

	tab->addTab(optionsTab, "Options");

	QVBoxLayout* paramsLayout = new QVBoxLayout;
	paramsLayout->addWidget(fpsCounter);
	paramsLayout->addWidget(tab);
	paramsLayout->addStretch();

	m_params->setLayout(paramsLayout);

	// ############ OPTION TAB - RENDERING MODE GROUPBOX ############

	QRadioButton* fill      = new QRadioButton("Fill");
	QRadioButton* wireframe = new QRadioButton("Wire Frame");
	QRadioButton* points    = new QRadioButton("Points");

	fill->setChecked(true);

	QVBoxLayout* renderingModeLayout = new QVBoxLayout;
	renderingModeLayout->addWidget(fill);
	renderingModeLayout->addWidget(wireframe);
	renderingModeLayout->addWidget(points);

	QGroupBox* renderingModeGroupBox = new QGroupBox("Rendering mode");
	renderingModeGroupBox->setLayout(renderingModeLayout);

	// ############ OPTION TAB - PROJECTION TYPE GROUPBOX ############

	QRadioButton* perspective = new QRadioButton("Perspective");
	QRadioButton* orthographic = new QRadioButton("Orthographic");

	perspective->setChecked(true);

	QVBoxLayout* projectionTypeLayout = new QVBoxLayout;
	projectionTypeLayout->addWidget(perspective);
	projectionTypeLayout->addWidget(orthographic);

	QGroupBox* projectionTypeGroupBox = new QGroupBox("Projection Type");
	projectionTypeGroupBox->setLayout(projectionTypeLayout);

	// ############ OPTION TAB - VIEW GROUPBOX ############

	fovLabel         = new QLabel("FOV :");
	nearPlaneLabel   = new QLabel("Near plane :");
	farPlaneLabel    = new QLabel("Far plane :");
	leftLabel        = new QLabel("Left :");
	rightLabel       = new QLabel("Right :");
	bottomLabel      = new QLabel("Bottom :");
	topLabel         = new QLabel("Top :");

	fovValue         = new QDoubleSpinBox;
	nearPlaneValue   = new QDoubleSpinBox;
	farPlaneValue    = new QDoubleSpinBox;
	leftValue        = new QDoubleSpinBox;
	rightValue       = new QDoubleSpinBox;
	bottomValue      = new QDoubleSpinBox;
	topValue         = new QDoubleSpinBox;

	fovValue->setRange(25.0, 130.0);
	fovValue->setValue(m_canvas->getScene()->getCamera()->fieldOfView());

	nearPlaneValue->setMinimum(0.01);
	nearPlaneValue->setSingleStep(0.01);
	nearPlaneValue->setValue(m_canvas->getScene()->getCamera()->nearPlane());

	farPlaneValue->setRange(0.001, 50000.0);
	farPlaneValue->setValue(m_canvas->getScene()->getCamera()->farPlane());

	leftLabel->hide();
	rightLabel->hide();
	bottomLabel->hide();
	topLabel->hide();

	leftValue->hide();
	rightValue->hide();
	bottomValue->hide();
	topValue->hide();

	leftValue->setRange(-200.0, 0.0);
	leftValue->setSingleStep(0.5);
	leftValue->setValue(m_canvas->getScene()->getCamera()->left());

	rightValue->setRange(0.0, 200.0);
	rightValue->setSingleStep(0.5);
	rightValue->setValue(m_canvas->getScene()->getCamera()->right());

	bottomValue->setRange(-200.0, 0.0);
	bottomValue->setSingleStep(0.5);
	bottomValue->setValue(m_canvas->getScene()->getCamera()->bottom());

	topValue->setRange(0.0, 200.0);
	topValue->setSingleStep(0.5);
	topValue->setValue(m_canvas->getScene()->getCamera()->top());

	QGridLayout* viewLayout = new QGridLayout;
	viewLayout->addWidget(fovLabel, 0, 0);
	viewLayout->addWidget(fovValue, 0, 1);

	viewLayout->addWidget(nearPlaneLabel, 1, 0);
	viewLayout->addWidget(nearPlaneValue, 1, 1);

	viewLayout->addWidget(farPlaneLabel, 2, 0);
	viewLayout->addWidget(farPlaneValue, 2, 1);

	viewLayout->addWidget(leftLabel, 3, 0);
	viewLayout->addWidget(leftValue, 3, 1);

	viewLayout->addWidget(rightLabel, 4, 0);
	viewLayout->addWidget(rightValue, 4, 1);

	viewLayout->addWidget(bottomLabel, 5, 0);
	viewLayout->addWidget(bottomValue, 5, 1);

	viewLayout->addWidget(topLabel, 6, 0);
	viewLayout->addWidget(topValue, 6, 1);

	QGroupBox* viewGroupBox = new QGroupBox("View");
	viewGroupBox->setLayout(viewLayout);

	// ############ OPTION TAB - CAMERA GROUPBOX ############

	QPushButton* resetCamera = new QPushButton("Reset camera");

	QRadioButton* firstPerson = new QRadioButton("First person");
	QRadioButton* thirdPerson = new QRadioButton("Third person");

	thirdPerson->setChecked(true);

	QFrame* hLineCamera1 = new QFrame;
	QFrame* hLineCamera2 = new QFrame;

	hLineCamera1->setFrameStyle(QFrame::HLine | QFrame::Sunken);
	hLineCamera2->setFrameStyle(QFrame::HLine | QFrame::Sunken);

	QLabel* cameraSpeedLabel       = new QLabel("Speed (m/s) : ");
	QLabel* cameraSensitivityLabel = new QLabel("Sensitivity : ");

	QDoubleSpinBox* cameraSpeedValue       = new QDoubleSpinBox;
	QDoubleSpinBox* cameraSensitivityValue = new QDoubleSpinBox;

	cameraSpeedValue->setRange(1.0, 50000.0);
	cameraSpeedValue->setValue(m_scene->getCamera()->speed());
	cameraSpeedValue->setMaximumSize(60, 20);

	cameraSensitivityValue->setValue(m_scene->getCamera()->sensitivity());
	cameraSensitivityValue->setSingleStep(0.05);
	cameraSensitivityValue->setMaximumSize(60, 20);

	QHBoxLayout* cameraSpeedLayout = new QHBoxLayout;
	cameraSpeedLayout->addWidget(cameraSpeedLabel);
	cameraSpeedLayout->addWidget(cameraSpeedValue);

	QHBoxLayout* cameraSensitivityLayout = new QHBoxLayout;
	cameraSensitivityLayout->addWidget(cameraSensitivityLabel);
	cameraSensitivityLayout->addWidget(cameraSensitivityValue);

	QVBoxLayout* cameraLayout = new QVBoxLayout;
	cameraLayout->addWidget(firstPerson);
	cameraLayout->addWidget(thirdPerson);
	cameraLayout->addWidget(hLineCamera1);
	cameraLayout->addLayout(cameraSpeedLayout);
	cameraLayout->addLayout(cameraSensitivityLayout);
	cameraLayout->addWidget(hLineCamera2);
	cameraLayout->addWidget(resetCamera);

	QGroupBox* cameraGroupBox = new QGroupBox("Camera");
	cameraGroupBox->setLayout(cameraLayout);

	// ###########################################################

	QVBoxLayout* optionsTabLayout = new QVBoxLayout;
	optionsTabLayout->addWidget(renderingModeGroupBox);
	optionsTabLayout->addWidget(projectionTypeGroupBox);
	optionsTabLayout->addWidget(viewGroupBox);
	optionsTabLayout->addWidget(cameraGroupBox);
	optionsTabLayout->addStretch();

	optionsTab->setLayout(optionsTabLayout);

	// ############ SIGNALS/SLOTS ############


	// Rendering mode
	connect(fill,      SIGNAL(toggled(bool)), m_scene, SLOT(toggleFill(bool)));
	connect(wireframe, SIGNAL(toggled(bool)), m_scene, SLOT(toggleWireframe(bool)));
	connect(points,    SIGNAL(toggled(bool)), m_scene, SLOT(togglePoints(bool)));

	// Projection type
	connect(perspective, SIGNAL(toggled(bool)), this, SLOT(setViewProperties(bool)));

	// View
	connect(fovValue,       SIGNAL(valueChanged(double)), this, SLOT(updateFieldOfView(double)));
	connect(nearPlaneValue, SIGNAL(valueChanged(double)), this, SLOT(updateNearPlane(double)));
	connect(farPlaneValue,  SIGNAL(valueChanged(double)), this, SLOT(updateFarPlane(double)));
	connect(leftValue,      SIGNAL(valueChanged(double)), this, SLOT(updateLeft(double)));
	connect(rightValue,     SIGNAL(valueChanged(double)), this, SLOT(updateRight(double)));
	connect(bottomValue,    SIGNAL(valueChanged(double)), this, SLOT(updateBottom(double)));
	connect(topValue,       SIGNAL(valueChanged(double)), this, SLOT(updateTop(double)));

	// Camera
	connect(cameraSpeedValue,       SIGNAL(valueChanged(double)), m_canvas.data(), SLOT(setCameraSpeed(double)));
	connect(cameraSensitivityValue, SIGNAL(valueChanged(double)), m_canvas.data(), SLOT(setCameraSensitivity(double)));
	connect(resetCamera,            SIGNAL(clicked()),            m_camera,            SLOT(resetCamera()));
	connect(firstPerson,            SIGNAL(toggled(bool)),        m_camera,            SLOT(switchToFirstPersonCamera(bool)));
	connect(thirdPerson,            SIGNAL(toggled(bool)),        m_camera,            SLOT(switchToThirdPersonCamera(bool)));

	// Update frame rate
	connect(m_canvas.data(), SIGNAL(updateFramerate()), this, SLOT(setFramerate()));

}


void MainWindow::setViewProperties(bool state)
{
	if(state)
	{
		m_camera->setProjectionType(Camera::Perspective);
		m_scene->resize(m_canvas->width(), m_canvas->height());

		fovLabel->show();
		fovValue->show();

		leftLabel->hide();
		rightLabel->hide();
		bottomLabel->hide();
		topLabel->hide();

		leftValue->hide();
		rightValue->hide();
		bottomValue->hide();
		topValue->hide();
	}
	else
	{
		m_camera->setProjectionType(Camera::Orthogonal);
		m_scene->resize(m_canvas->width(), m_canvas->height());

		fovLabel->hide();
		fovValue->hide();

		leftLabel->show();
		rightLabel->show();
		bottomLabel->show();
		topLabel->show();

		leftValue->show();
		rightValue->show();
		bottomValue->show();
		topValue->show();
	}
}

void MainWindow::setFullScreen(bool state)
{
	(state) ? showFullScreen() : showNormal();
}

void MainWindow::setFramerate()
{
	static double currentTime = 0;
	static double lastTime    = 0;
	static double average     = 0;

	static int count = 0;

	lastTime = currentTime;

	QTime time;
	currentTime = time.currentTime().msec();

	if(currentTime > lastTime)
	{
		average += 1000.0 / (currentTime - lastTime);
		count++;
	}

	if(count == 30)
	{
		fpsCounter->display(average/count);

		count   = 0;
		average = 0;
	}
}

void MainWindow::updateFieldOfView(double fov)
{
	m_camera->setFieldOfView(static_cast<float>(fov));
}

void MainWindow::updateNearPlane(double nearPlane)
{
	m_camera->setNearPlane(static_cast<float>(nearPlane));
}

void MainWindow::updateFarPlane(double farPlane)
{
	m_camera->setFarPlane(static_cast<float>(farPlane));
}

void MainWindow::updateLeft(double left)
{
	m_camera->setLeft(static_cast<float>(left));
}

void MainWindow::updateRight(double right)
{
	m_camera->setRight(static_cast<float>(right));
}

void MainWindow::updateBottom(double bottom)
{
	m_camera->setBottom(static_cast<float>(bottom));
}

void MainWindow::updateTop(double top)
{
	m_camera->setTop(static_cast<float>(top));
}

void MainWindow::showMessage( QtMsgType type, const QMessageLogContext &context, const QString &msg )
{
	statusBar()->showMessage(msg);
}

void MainWindow::showSystemLog()
{
	QDesktopServices::openUrl(QUrl::fromLocalFile("./Log.log"));
}

void MainWindow::showBackGroundColorPicker()
{
	QColor col = QColorDialog::getColor();
	if(col.isValid()) m_scene->setBackGroundColor(col);
}

void MainWindow::showSkyboxDialog()
{
	SkyboxDialog* dilog = new SkyboxDialog(m_scene->getSkybox(), this);
	dilog->show();
}

void MainWindow::actionTriggered( QAction* action )
{
	m_scene->objectManager()->setLoadingFlag(action->text());
}
