#pragma once
#include <QOpenGLFunctions_4_3_Core>
#include <Scene/AbstractModel.h>
#include <Scene/ShadingTechniques/SkyboxTechnique.h>
#include <Scene/Managers/MeshManager.h>
#include <Scene/Camera.h>
#include <Primitives/CubeMapTexture.h>
#include <Utility/ModelLoader.h>
class Scene;

class Skybox : public AbstractModel, protected QOpenGLFunctions_4_3_Core
{
public:
	Skybox(Scene* scene);
	~Skybox();

	bool init(const QString& PosXFilename,
			  const QString& NegXFilename,
			  const QString& PosYFilename,
			  const QString& NegYFilename,
			  const QString& PosZFilename,
			  const QString& NegZFilename);

	virtual void render( const float currentTime );

	CubemapTexturePtr getCubemapTexture() const { return m_cubemapTex; }

private:    
	void drawElements(uint index);

	GLuint m_vao;
	SkyboxTechniquePtr m_skyboxTechnique;
	Scene* m_scene;
	CubemapTexturePtr m_cubemapTex;
	QVector<MeshPtr> m_meshes;
};

typedef QSharedPointer<Skybox> SkyboxPtr;
