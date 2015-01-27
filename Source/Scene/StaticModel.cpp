#include "StaticModel.h"
#include <Scene/Scene.h>

StaticModel::StaticModel(const QString& name, Scene* scene, ShadingTechniquePtr tech)
  : AbstractModel(name),
	m_scene(scene),
    m_RenderingEffect(tech),
	m_vao(tech->getVAO())
{
	initialize();
}

StaticModel::StaticModel(const QString& name, Scene* scene, ShadingTechniquePtr tech, QVector<ModelDataPtr> modelData)
  : AbstractModel(name),
    m_scene(scene),
    m_RenderingEffect(tech),
	m_vao(tech->getVAO())
{
	initialize(modelData);
}


StaticModel::~StaticModel() 
{
	// clean up the textures (this always takes a lot of memory)
	foreach(QVector<TexturePtr> texVec, m_textures)
	{
		foreach(TexturePtr tex, texVec)
		{
			// erase it from the texture manager
			m_textureManager->deleteTexture(tex);

			// clear it
			tex.clear();
		}
	}

	// clean up the meshes
	foreach(MeshPtr mesh, m_meshes)
	{
		m_meshManager->deleteMesh(mesh);
		mesh.clear();
	}

	// clean up the materials
	foreach(MaterialPtr mat, m_materials)
	{
		m_materialManager->deleteMaterial(mat);
		mat.clear();
	}
}

void StaticModel::initRenderingEffect()
{ 	

	DirectionalLight directionalLight;
	directionalLight.Color = vec3(1.0f, 1.0f, 1.0f);
	directionalLight.AmbientIntensity = 0.55f;
	directionalLight.DiffuseIntensity = 0.9f;
	directionalLight.Direction = vec3(-1.0f, -1.0, 1.0);

	m_RenderingEffect->enable();
	m_RenderingEffect->setColorTextureUnit(0);
	m_RenderingEffect->setNormalMapTextureUnit(2);
	m_RenderingEffect->setDirectionalLight(directionalLight);
	m_RenderingEffect->setMatSpecularIntensity(0.0f);
	m_RenderingEffect->setMatSpecularPower(0);
	m_RenderingEffect->disable();
}


void StaticModel::initialize(QVector<ModelDataPtr> modelDataVector)
{
	Q_ASSERT(initializeOpenGLFunctions());

	initRenderingEffect();

	m_meshManager     = m_scene->meshManager();
	m_textureManager  = m_scene->textureManager();
	m_materialManager = m_scene->materialManager();

	// traverse modelData vector
	m_textures.resize(modelDataVector.size());
	for (int i = 0; i < modelDataVector.size(); ++i)
	{
		ModelDataPtr data = modelDataVector[i];

		// deal with the mesh
		MeshPtr mesh = m_meshManager->getMesh(data->meshData.name);
		if (!mesh)
		{
			mesh = m_meshManager->addMesh(data->meshData.name, data->meshData.numIndices, data->meshData.baseVertex, 	data->meshData.baseIndex);
		}

		m_meshes.push_back(mesh);

		// deal with the texture
		if(data->textureData.hasTexture)
		{
			TexturePtr  texture_colorMap = m_textureManager->getTexture(data->textureData.colorMap);
			if(!texture_colorMap)
			{
				texture_colorMap = m_textureManager->addTexture(data->textureData.colorMap, data->textureData.colorMap);
			}
			m_textures[i].push_back(texture_colorMap);

			if (!data->textureData.normalMap.isEmpty())
			{
				TexturePtr  texture_normalMap = m_textureManager->getTexture(data->textureData.normalMap);
				if(!texture_normalMap)
				{
					texture_normalMap = m_textureManager->addTexture(data->textureData.normalMap, data->textureData.normalMap, Texture::Texture2D, Texture::NormalMap);
				}
				m_textures[i].push_back(texture_normalMap);
			}
			
		}
		else m_textures[i].push_back(TexturePtr(nullptr));

		// deal with the material
		MaterialPtr material = m_materialManager->getMaterial(data->materialData.name);
		if(!material)
		{
			material = m_materialManager->addMaterial(data->materialData.name,
													data->materialData.ambientColor,
													data->materialData.diffuseColor,
													data->materialData.specularColor,
													data->materialData.emissiveColor,
													data->materialData.shininess,
													data->materialData.shininessStrength,
													data->materialData.twoSided,
													data->materialData.blendMode,
													data->materialData.alphaBlending,
													data->textureData.hasTexture);
		}

		m_materials.push_back(material);

	}

}

void StaticModel::destroy() {}

void StaticModel::render( float time )
{
	m_RenderingEffect->enable();

	QMatrix4x4 modelMatrix = m_actor->getTranformMatrix();
	
	//QMatrix3x3 normalMatrix = modelViewMatrix.normalMatrix();
	m_RenderingEffect->setEyeWorldPos(m_scene->getCamera()->position());
	m_RenderingEffect->setWVP(m_scene->getCamera()->viewProjectionMatrix() * modelMatrix);
	m_RenderingEffect->setWorldMatrix(modelMatrix); 

	for(int i = 0; i < m_meshes.size(); ++i)
	{
		/*if( m_materials[i] != nullptr && ! m_materials[i]->isTranslucent())*/
		{
			for(int j = 0; j < m_textures[i].size(); ++j)
			{
				TexturePtr pTexture = m_textures[i][j];
				if(pTexture)
				{
					if (pTexture->usage() == Texture::ColorMap)
					{
						pTexture->bind(COLOR_TEXTURE_UNIT);
					}
					else if (pTexture->usage() == Texture::NormalMap)
					{
						pTexture->bind(NORMAL_TEXTURE_UNIT);
					}
				}
			}
			

			//m_materials[i]->bind();

			drawElements(i, BaseVertex);
		}
	}

// 	for(int i = 0; i < m_meshes.size(); ++i)
// 	{
// 		if( m_materials[i] != nullptr && m_materials[i]->isTranslucent())
// 		{
// 			glDepthMask(GL_FALSE);
// 			glEnable(GL_BLEND);
// 
// 			m_materials[i]->bind();
// 
// 			drawElements(i, Indexed | BaseVertex);
// 
// 			glDisable(GL_BLEND);
// 			glDepthMask(GL_TRUE);
// 		}
// 	}

	for (int i = 0; i < m_textures.size(); ++i)
	{
		for(int j = 0; j < m_textures[i].size(); ++j)
		{
			TexturePtr pTexture = m_textures[i][j];
			if(pTexture)
			{
				pTexture->release();
			}
		}
	}
	

	m_RenderingEffect->disable();	
}

void StaticModel::drawElements(unsigned int index, int mode)
{
	// Mode has not been implemented yet
	Q_UNUSED(mode);
	glBindVertexArray(m_vao);

	glDrawElementsBaseVertex(
		GL_TRIANGLES,
		m_meshes[index]->getNumIndices(),
		GL_UNSIGNED_INT,
		reinterpret_cast<void*>((sizeof(unsigned int)) * m_meshes[index]->getBaseIndex()),
		m_meshes[index]->getBaseVertex()
		);

	// Make sure the VAO is not changed from the outside    
	glBindVertexArray(0);
}