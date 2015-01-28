#pragma once
#include <QObject>
#include <QOpenGLContext>
#include <QElapsedTimer>

class AbstractScene : public QObject
{
public:
	AbstractScene(QObject* parent = 0);

    /**
     * Loading shaders, VBOs, IBOs, textures etc ...
     */
    virtual void initialize() = 0;

    /**
     * Update the scene
     */
    virtual void update(float t) = 0;

    /**
     * Resize
     */
    virtual void resize(int width, int height) = 0;

    /**
     * Access methods to the context of the OpenGL scene
     */
    inline void setContext(QOpenGLContext* context) { m_context = context; }
    inline QOpenGLContext* context() const { return m_context; }

	/**
     * Associate with the canvas widget
     */
	void setCanvas(QObject* widget) { m_canvas = widget; }

protected:
    QOpenGLContext* m_context;
	QObject* m_canvas;
};