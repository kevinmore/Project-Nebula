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
	vec3 getRayFromMouse(const QPoint& mousePos);
	bool testRayOBBIntersection(const vec3& rayDirection,
								const GameObjectPtr target, float& intersectionDistance);
	bool testRaySpehreIntersection(const vec3& rayDirection, const float radius, 
								const GameObjectPtr target, float& intersectionDistance);

	GameObjectPtr pickObject(const QPoint& mousePos);

public slots:
	void setCameraSpeed(double speed);
	void setCameraSensitivity(double sensitivity);
	void showGPUInfo();

protected slots:
	void resizeGL();
	void paintGL();
	void updateScene();

signals:
	void updateFramerate();
	void objectPicked(const QString& name);

private:
	QScopedPointer<QOpenGLContext> m_context;
	AbstractScene* m_scene;

	QElapsedTimer m_updateTimer;

	QPoint m_prevPos;
	QPoint m_pos;

	bool m_rightButtonPressed;
	bool m_middleButtonPressed;

	double m_cameraSpeed;
	double m_cameraSensitivity;
};

