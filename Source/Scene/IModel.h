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

#include <Physicis/Collision/Collider/SphereCollider.h>
#include <Physicis/Collision/Collider/BoxCollider.h>
#include <Physicis/Collision/Collider/ConvexHullCollider.h>


class IModel : public Component, protected QOpenGLFunctions_4_3_Core
{
public:
	IModel(ShadingTechniquePtr tech = ShadingTechniquePtr(), const QString& fileName = "");
	virtual ~IModel() = 0;

	virtual void render(const float currentTime) = 0;
	virtual QString className() { return "Model"; }

	QString fileName() const { return m_fileName; }
	void setFileName(QString& file) { m_fileName = file; }

	ShadingTechniquePtr renderingEffect() const { return m_renderingEffect; }
	QVector<MaterialPtr> getMaterials() const { return m_materials; }

	void setBoundingBox(BoxCollider* box);
	BoxColliderPtr getBoundingBox() const;
	void setBoundingSphere(SphereCollider* sphere);
	SphereColliderPtr getBoundingSphere() const;
	void setConvexHullCollider(ConvexHullCollider* ch);
	ConvexHullColliderPtr getConvexHullCollider() const;

	void showBoundingVolume();
	void hideBoundingVolume();

	inline const mat4& getTransformMatrix() const { return m_transformMatrix; }

	enum PolygonMode
	{
		Fill  = GL_FILL,
		Line  = GL_LINE,
		Point = GL_POINT
	};

	void setPolygonMode(PolygonMode mode);

	const vec3& getScale() const { return m_scale; }

protected:
	virtual void syncTransform(const Transform& transform);
	void init();
	void drawElements(unsigned int index);

	QString m_fileName;
	ShadingTechniquePtr m_renderingEffect;
	QVector<MeshPtr> m_meshes;
	QVector<MaterialPtr> m_materials;
	GLuint m_vao;
	PolygonMode m_polygonMode;

	ColliderPtr m_currentBoundingVolume;
	SphereColliderPtr m_boundingSpehre;
	BoxColliderPtr m_boundingBox;
	ConvexHullColliderPtr m_convexHull;

	mat4 m_transformMatrix;
	vec3 m_scale;
};