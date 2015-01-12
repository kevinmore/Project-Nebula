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
	void initializeParamsArea();
	StateMachineViewer* showStateMachine(QStateMachine* machine);

private slots:
	void setFullScreen(bool state);
	void setViewProperties(bool state);
	void setFramerate();
	void updateMatrix();
	void updateFieldOfView(double fov);
	void updateNearPlane(double nearPlane);
	void updateFarPlane(double farPlane);
	void updateLeft(double left);
	void updateRight(double right);
	void updateBottom(double bottom);
	void updateTop(double top);

private:
	QScopedPointer<Canvas> m_canvas;

	QWidget      * m_params,         * m_coordinate,     * m_mvpMatrix;
	QDockWidget  * m_dockParamsArea, * m_dockMatrixArea, * m_stateMachineViewer, * m_heirarchyViewer;
	Scene        * m_scene;
	GameObject     * m_object3D;
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

	QLabel * modelMatrix00, * modelMatrix01, * modelMatrix02, * modelMatrix03,
		   * modelMatrix10, * modelMatrix11, * modelMatrix12, * modelMatrix13,
		   * modelMatrix20, * modelMatrix21, * modelMatrix22, * modelMatrix23,
		   * modelMatrix30, * modelMatrix31, * modelMatrix32, * modelMatrix33;

	QLabel * viewMatrix00, * viewMatrix01, * viewMatrix02, * viewMatrix03,
		   * viewMatrix10, * viewMatrix11, * viewMatrix12, * viewMatrix13,
		   * viewMatrix20, * viewMatrix21, * viewMatrix22, * viewMatrix23,
		   * viewMatrix30, * viewMatrix31, * viewMatrix32, * viewMatrix33;

	QLabel * projectionMatrix00, * projectionMatrix01, * projectionMatrix02, * projectionMatrix03,
		   * projectionMatrix10, * projectionMatrix11, * projectionMatrix12, * projectionMatrix13,
		   * projectionMatrix20, * projectionMatrix21, * projectionMatrix22, * projectionMatrix23,
		   * projectionMatrix30, * projectionMatrix31, * projectionMatrix32, * projectionMatrix33;
};

