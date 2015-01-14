#include "RiggedModel.h"
#include <Scene/Scene.h>
#include <QOpenGLContext>
#include <Animation/Rig/Pose.h>

RiggedModel::RiggedModel(const QString& name, Scene* scene, ShadingTechnique* tech, Skeleton* skeleton)
  : AbstractModel(name),
    m_scene(scene),
    m_RenderingEffect(tech),
	m_vao(tech->getVAO()),
	m_skeleton(skeleton),
	m_FKController(0),
	m_IKSolver(0),
	m_hasAnimation(false),
	m_animationDuration(0.0f)
{
	initialize();
}

RiggedModel::RiggedModel(const QString& name, Scene* scene, ShadingTechnique* tech, Skeleton* skeleton, QVector<ModelDataPtr> modelData)
  : AbstractModel(name),
    m_scene(scene),
    m_RenderingEffect(tech),
	m_vao(tech->getVAO()),
	m_skeleton(skeleton),
	m_FKController(0),
	m_IKSolver(0),
	m_hasAnimation(false),
	m_animationDuration(0.0f)
{
	initialize(modelData);
}


RiggedModel::~RiggedModel() 
{
	SAFE_DELETE(m_RenderingEffect);
 	SAFE_DELETE(m_skeleton);
 	SAFE_DELETE(m_IKSolver);
	SAFE_DELETE(m_FKController);
}

void RiggedModel::initRenderingEffect()
{ 	

	DirectionalLight directionalLight;
	directionalLight.Color = vec3(1.0f, 1.0f, 1.0f);
	directionalLight.AmbientIntensity = 0.55f;
	directionalLight.DiffuseIntensity = 0.9f;
	directionalLight.Direction = vec3(-1.0f, 0.0, -1.0);

	m_RenderingEffect->Enable();
	m_RenderingEffect->SetColorTextureUnit(0);
	m_RenderingEffect->SetNormalMapTextureUnit(2);
	m_RenderingEffect->SetDirectionalLight(directionalLight);
	m_RenderingEffect->SetMatSpecularIntensity(0.0f);
	m_RenderingEffect->SetMatSpecularPower(0);
	m_RenderingEffect->Disable();
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
	m_textures.resize(modelDataVector.size());
	for (int i = 0; i < modelDataVector.size(); ++i)
	{
		ModelDataPtr data = modelDataVector[i];

		m_hasAnimation = data->hasAnimation;
		if(m_hasAnimation) m_animationDuration = data->animationDuration;

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


// 	ikSolved = false;
// 	lastUpdatedTime = 0.0f;
// 	updateIKRate = 0.5f;
// 
// 	m_FABRSolver = new FABRIKSolver(m_skeleton, 0.1f);
// 	m_FABRSolver->enableIKChain("Bip01_L_UpperArm", "Bip01_L_Hand");
// 
// 
// 	// set bone DOFs
// 	Bone* pBone;
// 	Bone::AngleLimits pitchConstraint, yawConstraint, rollConstraint;
// 	Bone::DimensionOfFreedom dof;
// 	Math::EulerAngle ea;
// 	
// 
// 	pBone = m_skeleton->getBone("Bip01_L_Clavicle");
// 	ea = pBone->getEulerAnglesInModelSpace();
// 	pitchConstraint = Bone::AngleLimits(ea.m_fPitch, ea.m_fPitch);
// 	yawConstraint = Bone::AngleLimits(ea.m_fYaw - 20.0f, ea.m_fYaw + 20.0f);
// 	rollConstraint = Bone::AngleLimits(ea.m_fRoll - 20.0f, ea.m_fRoll + 20.0f);
// 	dof = Bone::DimensionOfFreedom(pitchConstraint, yawConstraint, rollConstraint);
// 	pBone->setDof(dof);
// 
// 	pBone = m_skeleton->getBone("Bip01_L_UpperArm");
// 	ea = pBone->getEulerAnglesInModelSpace();
// 	pitchConstraint = Bone::AngleLimits(ea.m_fPitch - 50.0f, ea.m_fPitch + 50.0f);
// 	yawConstraint = Bone::AngleLimits(ea.m_fYaw - 120.0f, ea.m_fYaw + 30.0f);
// 	rollConstraint = Bone::AngleLimits(ea.m_fRoll - 170.0f, ea.m_fRoll + 40.0f);
// 	dof = Bone::DimensionOfFreedom(pitchConstraint, yawConstraint, rollConstraint);
// 	pBone->setDof(dof);
// 	
// 	pBone = m_skeleton->getBone("Bip01_L_Forearm");
// 	ea = pBone->getEulerAnglesInModelSpace();
// 	pitchConstraint = Bone::AngleLimits(ea.m_fPitch - 5.0f, ea.m_fPitch + 120.0f);
// 	yawConstraint = Bone::AngleLimits(ea.m_fYaw - 60.0f, ea.m_fYaw + 20.0f);
// 	rollConstraint = Bone::AngleLimits(ea.m_fRoll, ea.m_fRoll);
// 	dof = Bone::DimensionOfFreedom(pitchConstraint, yawConstraint, rollConstraint);
// 	pBone->setDof(dof);
// 
// 	pBone = m_skeleton->getBone("Bip01_L_Hand");
// 	ea = pBone->getEulerAnglesInModelSpace();
// 	pitchConstraint = Bone::AngleLimits(ea.m_fPitch - 10.0f, ea.m_fPitch + 10.0f);
// 	yawConstraint = Bone::AngleLimits(ea.m_fYaw - 120.0f, ea.m_fYaw + 30.0f);
// 	rollConstraint = Bone::AngleLimits(ea.m_fRoll - 5.0f, ea.m_fRoll + 5.0f);
// 	dof = Bone::DimensionOfFreedom(pitchConstraint, yawConstraint, rollConstraint);
// 	pBone->setDof(dof);
// 
// 	pBone = m_skeleton->getBone("Bip01_L_Finger2");
// 	pitchConstraint = Bone::AngleLimits(ea.m_fPitch, ea.m_fPitch);
// 	yawConstraint = Bone::AngleLimits(ea.m_fYaw, ea.m_fYaw);
// 	rollConstraint = Bone::AngleLimits(ea.m_fRoll - 5.0f, ea.m_fRoll + 90.0f);
// 	dof = Bone::DimensionOfFreedom(pitchConstraint, yawConstraint, rollConstraint);
// 	pBone->setDof(dof);
// 
// 	pBone = m_skeleton->getBone("Bip01_L_Finger21");
// 	ea = pBone->getEulerAnglesInModelSpace();
// 	pitchConstraint = Bone::AngleLimits(ea.m_fPitch, ea.m_fPitch);
// 	yawConstraint = Bone::AngleLimits(ea.m_fYaw, ea.m_fYaw);
// 	rollConstraint = Bone::AngleLimits(ea.m_fRoll - 5.0f, ea.m_fRoll + 90.0f);
// 	dof = Bone::DimensionOfFreedom(pitchConstraint, yawConstraint, rollConstraint);
// 	pBone->setDof(dof);
// 
// 	solvingDuration= 1.0f;
}

void RiggedModel::destroy() {}

void RiggedModel::render( float currentTime )
{
	float dt = currentTime - lastUpdatedTime;
	lastUpdatedTime = currentTime;
	if(dt < 0)
	{
		dt = 0.0f;
	}
	//m_actor->translateInWorld(m_actor->globalSpeed() * dt); // this is for inplace locamotion
	
	m_RenderingEffect->Enable();

	QMatrix4x4 modelMatrix = m_actor->modelMatrix();
	modelMatrix.rotate(90, Math::Vector3D::UNIT_X); // this is for dae files
	QMatrix4x4 modelViewMatrix = m_scene->getCamera()->viewMatrix() * modelMatrix;
	QMatrix3x3 normalMatrix = modelViewMatrix.normalMatrix();

	m_RenderingEffect->SetEyeWorldPos(m_scene->getCamera()->position());
	m_RenderingEffect->SetWVP(m_scene->getCamera()->projectionMatrix() * modelViewMatrix);
	m_RenderingEffect->SetWorldMatrix(modelMatrix); 


	// do the skeleton animation here
	// check if the model has animation first
	QVector<QMatrix4x4> Transforms;
	m_FKController->getBoneTransforms(currentTime, Transforms);

	/*
	// use IK
	// CCD
	// set constraint
	CCDIKSolver::IkConstraint constraint;
	constraint.m_startBone = m_skeleton->getBone("Bip01_L_Clavicle");
	constraint.m_endBone = m_skeleton->getBone("Bip01_L_Finger22");
	constraint.m_targetMS = m_targetPos;
	
	if (m_CCDSolver->solveOneConstraint( constraint, m_skeleton ))
	{
		m_FKController->disableBoneChain(m_skeleton->getBone("Bip01_L_Clavicle"));
		m_FKController->getBoneTransforms(time, Transforms);
		m_CCDSolver->getBoneTransforms(m_skeleton, constraint.m_startBone, Transforms);
	}
	else
	{
		m_FKController->enableAllBones();
		m_FKController->getBoneTransforms(time, Transforms);
	}
	*/

 	// update the bone positions
	for (int i = 0 ; i < Transforms.size() ; ++i) 
	{
		m_RenderingEffect->SetBoneTransform(i, Transforms[i]);
	}



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

	m_RenderingEffect->Disable();
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

void RiggedModel::setReachableTargetPos( vec3& pos )
{
	m_targetPos = pos;
}

void RiggedModel::setFKController( FKController* fkCtrl )
{
	m_FKController = fkCtrl;
}

void RiggedModel::setIKSolver( CCDIKSolver* ikSolver )
{
	m_IKSolver = ikSolver;
}
