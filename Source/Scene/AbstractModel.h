#pragma once
#include <QString>
#include <QSharedPointer>
#include <QColor>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLShaderProgram>

#include <Scene/GameObject.h>
#include <Primitives/Component.h>
#include <Primitives/Material.h>
#include <Scene/ShadingTechniques/ShadingTechnique.h>

struct MeshData
{
	QString name;

	unsigned int numIndices;
	unsigned int baseVertex;
	unsigned int baseIndex;
};


struct TextureData
{
	QString colorMap, normalMap;
	bool hasTexture;
};

struct MaterialData
{
	QString name;

	QColor ambientColor;
	QColor diffuseColor;
	QColor specularColor;
	QColor emissiveColor;

	float shininess;
	float shininessStrength;

	int twoSided;
	int blendMode;
	bool alphaBlending;
};

struct ModelData
{
	MeshData     meshData;
	TextureData  textureData;
	MaterialData materialData;
	bool hasAnimation;
	float animationDuration;
};

typedef QSharedPointer<MeshData> MeshDataPtr;
typedef QSharedPointer<TextureData> TextureDataPtr;
typedef QSharedPointer<MaterialData> MaterialDataPtr;
typedef QSharedPointer<ModelData> ModelDataPtr;

class AbstractModel : public Component
{
public:
	AbstractModel(ShadingTechniquePtr tech = ShadingTechniquePtr(), const QString& fileName = "");
	virtual ~AbstractModel() = 0;

	virtual void render(const float currentTime) = 0;
	virtual QString className() { return "Model"; }

	QString fileName() const { return m_fileName; }
	void setFileName(QString& file) { m_fileName = file; }

	ShadingTechniquePtr renderingEffect() const { return m_RenderingEffect; }
	MaterialPtr getMaterial() { return m_materials[0]; }

protected:
	QString m_fileName;
	ShadingTechniquePtr m_RenderingEffect;
	QVector<MaterialPtr> m_materials;
};