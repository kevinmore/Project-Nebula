#pragma once
#include <QObject>
#include <QOpenGLContext>
#include <QElapsedTimer>
class Canvas;
class IScene : public QObject
{
public:
	IScene(QObject* parent = 0);

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
	void setCanvas(Canvas* widget) { m_canvas = widget; }
	Canvas* getCanvas() { return m_canvas; }

protected:
    QOpenGLContext* m_context;
	Canvas* m_canvas;
};