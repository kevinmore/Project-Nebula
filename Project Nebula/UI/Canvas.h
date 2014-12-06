#pragma once
#include <QWindow>
#include <QElapsedTimer>
#include <QOpenGLContext>
#include <Scene/Scene.h>
#include <QMouseEvent>
#include <QKeyEvent>
#include <QWheelEvent>

/************************************************************************/
/*This class is used to initialize the OpenGL context 
and manipulate the scene rendering.                                     */
/************************************************************************/
class Canvas : public QWindow
{

	Q_OBJECT


public:
	Canvas(QScreen* screen = 0);
	~Canvas();

	Scene* getScene();

protected:
	void keyPressEvent(QKeyEvent* e);
	void keyReleaseEvent(QKeyEvent* e);
	void mousePressEvent(QMouseEvent* e);
	void mouseReleaseEvent(QMouseEvent* e);
	void mouseMoveEvent(QMouseEvent* e);
	void wheelEvent(QWheelEvent *e);

private:
	void initializeGL();

public slots:
	void checkAnimate(int state);
	void setCameraSpeed(double speed);
	void setCameraSensitivity(double sensitivity);

protected slots:
	void resizeGL();
	void paintGL();
	void updateScene();

signals:
	void updateFramerate();

private:
	QScopedPointer<QOpenGLContext> m_context;
	AbstractScene* m_scene;

	QElapsedTimer m_renderTimer;
	QElapsedTimer m_updateTimer;

	QPoint m_prevPos;
	QPoint m_pos;

	bool m_rightButtonPressed;

	double m_cameraSpeed;
	double m_cameraSensitivity;
};

