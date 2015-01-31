#pragma once
#include <QString>
#include <QSharedPointer>
#include <QColor>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLShaderProgram>
#include <QOpenGLFunctions_4_3_Core>

#include <Primitives/GameObject.h>
#include <Primitives/Component.h>
#include <Primitives/Mesh.h>
#include <Primitives/Material.h>
#include <Scene/ShadingTechniques/ShadingTechnique.h>



class AbstractModel : public Component, protected QOpenGLFunctions_4_3_Core
{
public:
	AbstractModel(ShadingTechniquePtr tech = ShadingTechniquePtr(), const QString& fileName = "");
	virtual ~AbstractModel() = 0;

	virtual void render(const float currentTime) = 0;
	virtual QString className() { return "Model"; }

	QString fileName() const { return m_fileName; }
	void setFileName(QString& file) { m_fileName = file; }

	ShadingTechniquePtr renderingEffect() const { return m_RenderingEffect; }
	MaterialPtr getMaterial() const { return m_materials[0]; }


	enum PolygonMode
	{
		Fill  = GL_FILL,
		Line  = GL_LINE,
		Point = GL_POINT
	};

	void setPolygonMode(PolygonMode mode);

protected:

	void init();
	void drawElements(unsigned int index);

	QString m_fileName;
	ShadingTechniquePtr m_RenderingEffect;
	QVector<MeshPtr> m_meshes;
	QVector<MaterialPtr> m_materials;
	GLuint m_vao;
	PolygonMode m_polygonMode;
};