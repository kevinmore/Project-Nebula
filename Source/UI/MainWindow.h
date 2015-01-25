#pragma once
#include <QtWidgets>
#include <UI/Canvas.h>
#include <Scene/Scene.h>
#include <Scene/Camera.h>
#include <Scene/GameObject.h>
#include <statemachineviewer.h>

class MainWindow : public QMainWindow
{
	Q_OBJECT

public:
	MainWindow(QWidget* parent = 0);
	~MainWindow();

private:
	void initializeCanvas();
	void initializeMenuBar();
	void initializeRightDockableArea();
	void initializeParamsArea();
	StateMachineViewer* showStateMachine(QStateMachine* machine);

private slots:
	void setFullScreen(bool state);
	void setViewProperties(bool state);
	void setFramerate();
	void updateFieldOfView(double fov);
	void updateNearPlane(double nearPlane);
	void updateFarPlane(double farPlane);
	void updateLeft(double left);
	void updateRight(double right);
	void updateBottom(double bottom);
	void updateTop(double top);
	void showMessage(QtMsgType type, const QMessageLogContext &context, const QString &msg);
	void showSystemLog();
	void showBackGroundColorPicker();
	void showSkyboxDialog();
	void actionTriggered(QAction* action);

private:
	QScopedPointer<Canvas> m_canvas;

	QWidget      * m_params,         * m_coordinate,     * m_mvpMatrix;
	QDockWidget  * m_dockSettingsArea, * m_dockMatrixArea, * m_stateMachineViewer, * m_heirarchyViewer;
	Scene        * m_scene;
	Camera  * m_camera;

	QLabel * fovLabel,   * leftLabel,
		   * rightLabel, * bottomLabel,
		   * topLabel,   * nearPlaneLabel,
		   * farPlaneLabel;

	QDoubleSpinBox * fovValue,      * nearPlaneValue,
			       * farPlaneValue, * leftValue,
			       * rightValue,    * bottomValue,
			       * topValue;

	QLCDNumber* fpsCounter;
};