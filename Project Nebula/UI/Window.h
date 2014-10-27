#pragma once
#include <GL/glew.h>
#include <QtGui/QWindow>
#include <QtCore/QElapsedTimer>
#include <QtGui/QOpenGLContext>
#include <Scene/Scene.h>

/************************************************************************/
/*This class is used to initialize the OpenGL context 
and manipulate the scene rendering.                                     */
/************************************************************************/
class Window : public QWindow
{

	Q_OBJECT


public:
	Window(QScreen* screen = 0);
	~Window(void);

	Scene* getScene();

protected:
	void keyPressEvent(QKeyEvent* e);
	void keyReleaseEvent(QKeyEvent* e);
	void mousePressEvent(QMouseEvent* e);
	void mouseReleaseEvent(QMouseEvent* e);
	void mouseMoveEvent(QMouseEvent* e);

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

	bool m_leftButtonPressed;

	double m_cameraSpeed;
	double m_cameraSensitivity;
};

