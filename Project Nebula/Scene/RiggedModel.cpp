#include "RiggedModel.h"
#include <Scene/Scene.h>
#include <QtGui/QOpenGLContext>

RiggedModel::RiggedModel(Scene* scene, ShadingTechnique* tech, Skeleton* skeleton, FKController* fkCtrl, CCDIKSolver* ikSolver, const GLuint vao)
  : m_scene(scene),
    m_RenderingEffect(tech),
	m_vao(vao),
	m_skeleton(skeleton),
	m_FKController(fkCtrl),
	m_IKSolver(ikSolver),
	m_hasAnimation(false),
	m_actor(new Object3D)
{
	initialize();
}

RiggedModel::RiggedModel(Scene* scene, ShadingTechnique* tech, Skeleton* skeleton, FKController* fkCtrl, CCDIKSolver* ikSolver, const GLuint vao, QVector<ModelDataPtr> modelData)
  : m_scene(scene),
    m_RenderingEffect(tech),
	m_vao(vao),
	m_skeleton(skeleton),
	m_FKController(fkCtrl),
	m_IKSolver(ikSolver),
	m_hasAnimation(false),
	m_actor(new Object3D)
{
	initialize(modelData);
}


RiggedModel::~RiggedModel() 
{
}

void RiggedModel::initRenderingEffect()
{ 	
	m_funcs->glClearDepth( 1.0 );
	m_funcs->glClearColor(0.39f, 0.39f, 0.39f, 0.0f);
	m_funcs->glEnable(GL_DEPTH_TEST);
	m_funcs->glDepthFunc(GL_LEQUAL);
	//m_funcs->glEnable(GL_CULL_FACE);

	DirectionalLight directionalLight;
	directionalLight.Color = vec3(1.0f, 1.0f, 1.0f);
	directionalLight.AmbientIntensity = 0.55f;
	directionalLight.DiffuseIntensity = 0.9f;
	directionalLight.Direction = vec3(-1.0f, 0.0, -1.0);

	m_RenderingEffect->Enable();
	m_RenderingEffect->SetColorTextureUnit(0);
	m_RenderingEffect->SetDirectionalLight(directionalLight);
	m_RenderingEffect->SetMatSpecularIntensity(0.0f);
	m_RenderingEffect->SetMatSpecularPower(0);

}


void RiggedModel::initialize(QVector<ModelDataPtr> modelDataVector)
{
	QOpenGLContext* context = QOpenGLContext::currentContext();

	Q_ASSERT(context);

	m_funcs = context->versionFunctions<QOpenGLFunctions_4_3_Core>();
	m_funcs->initializeOpenGLFunctions();

	initRenderingEffect();

	m_meshManager     = m_scene->meshManager();
	m_textureManager  = m_scene->textureManager();
	m_materialManager = m_scene->materialManager();

	// traverse modelData vector
	for (int i = 0; i < modelDataVector.size(); ++i)
	{
		ModelDataPtr data = modelDataVector[i];

		m_hasAnimation = data->hasAnimation;

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
			TexturePtr  texture = m_textureManager->getTexture(data->textureData.filename);

			if(!texture)
			{
				texture = m_textureManager->addTexture(data->textureData.filename, data->textureData.filename);
			}

			m_textures.push_back(texture);
		}
		else m_textures.push_back(TexturePtr(nullptr));

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


	ikSolved = false;
	lastUpdatedTime = 0.0f;
	updateIKRate = 0.2f;
}

void RiggedModel::destroy() {}

void RiggedModel::render( float time )
{

	QMatrix4x4 modelMatrix = m_actor->modelMatrix();
	modelMatrix.rotate(180, Math::Vector3D::UNIT_X); // this is for dae files
	QMatrix4x4 modelViewMatrix = m_scene->getCamera()->viewMatrix() * modelMatrix;
	QMatrix3x3 normalMatrix = modelViewMatrix.normalMatrix();

	m_RenderingEffect->SetEyeWorldPos(m_scene->getCamera()->position());
	m_RenderingEffect->SetWVP(m_scene->getCamera()->projectionMatrix() * modelViewMatrix);
	m_RenderingEffect->SetWorldMatrix(modelMatrix); 


	// do the skeleton animation here
	// check if the model has animation first
	QVector<QMatrix4x4> Transforms;

	if(m_hasAnimation)
	{
 		m_RenderingEffect->getShader()->setUniformValue("hasAnimation", true);
 		//m_FKController->BoneTransform(time, Transforms);
	}

	// INIT IK
	// set constraint
	//if (time - lastUpdatedTime > updateIKRate)
	{
		m_skeleton->getBone("Bip01_R_Hand")->isXConstraint = true;

		CCDIKSolver::IkConstraint constraint;
		constraint.m_startBone = m_skeleton->getBone("Bip01_R_UpperArm");
		constraint.m_endBone = m_skeleton->getBone("Bip01_R_Hand");
		constraint.m_targetMS = vec3(vec3(5*qSin(time), 0, 0));
		//if (!ikSolved)
		{
			ikSolved = m_IKSolver->solveOneConstraint( constraint, m_skeleton );
		}

		m_IKSolver->BoneTransform(m_skeleton, constraint.m_startBone, constraint.m_endBone, Transforms);

		lastUpdatedTime = time;

	}
	


 	// update the bone positions
	for (int i = 0 ; i < Transforms.size() ; i++) {
		m_RenderingEffect->SetBoneTransform(i, Transforms[i]);
	}



	for(int i = 0; i < m_meshes.size(); ++i)
	{
		/*if( m_materials[i] != nullptr && ! m_materials[i]->isTranslucent())*/
		{
			if(m_textures[i] != nullptr)
			{
				m_textures[i]->bind(GL_TEXTURE0);
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


}

void RiggedModel::drawElements(unsigned int index, int mode)
{
	// Mode has not been implemented yet
	Q_UNUSED(mode);
	m_funcs->glBindVertexArray(m_vao);

	m_funcs->glDrawElementsBaseVertex(
		GL_TRIANGLES,
		m_meshes[index]->getNumIndices(),
		GL_UNSIGNED_INT,
		reinterpret_cast<void*>((sizeof(unsigned int)) * m_meshes[index]->getBaseIndex()),
		m_meshes[index]->getBaseVertex()
		);
	// Make sure the VAO is not changed from the outside    
	m_funcs->glBindVertexArray(0);
}