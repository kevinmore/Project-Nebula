#include <limits.h>
#include <assert.h>
#include "ShadingTechnique.h"
#include <QtMath>
#include <Scene/Scene.h>

ShadingTechnique::ShadingTechnique(const QString &shaderName, ShaderType shaderType, Scene* scene)
	: Technique(shaderName),
	  m_scene(scene),
	  m_shaderType(shaderType),
	  usingCubeMap(false)
{
	if (!init()) 
	{
		qWarning() << shaderName << "may not be initialized successfully.";
	}
}

void ShadingTechnique::enable()
{
	m_shaderProgram->bind();
	if (usingCubeMap && m_scene)
	{
		SkyboxPtr skybox = m_scene->getSkybox();
		if(!skybox) return;
		CubemapTexturePtr cubeMap = skybox->getCubemapTexture();
		cubeMap->bind(GL_TEXTURE0);
		m_shaderProgram->setUniformValue("drawSkyBox", true);
	}
	else
	{
		m_shaderProgram->setUniformValue("drawSkyBox", false);
	}
}

bool ShadingTechnique::init()
{
    if (!Technique::init() || m_shaderFileName.isEmpty()) 
	{
        return false;
    }

	// if it's not used by a collider
	if (m_scene)
	{
		connect(m_scene, SIGNAL(ligthsChanged()), this, SLOT(updateLights()));
	}

	return compileShader();
}

bool ShadingTechnique::compileShader()
{
	QString vertexShader = (m_shaderType == STATIC) ? "static.vert" : "skinning.vert";
	m_shaderProgram->addShaderFromSourceFile(QOpenGLShader::Vertex, m_shaderFilePath + vertexShader);
	m_shaderProgram->addShaderFromSourceFile(QOpenGLShader::Fragment, m_shaderFilePath + m_shaderFileName + ".frag");
	m_shaderProgram->link();

	if (m_shaderFileName.contains("reflection"))
		usingCubeMap = true;
	else
		usingCubeMap = false;

	// process the lights in the scene
	updateLights();

	return true;
}

void ShadingTechnique::updateLights()
{
	// get the lights from the scene
	if (!m_scene) return;
	QList<LightPtr> lights = m_scene->getLights();

	// process each light
	uint dircLightsCount, pointLightsCount, spotLightsCount;
	dircLightsCount = pointLightsCount = spotLightsCount = 0;

	m_shaderProgram->bind();

	foreach(LightPtr light, lights)
	{
		Light::LightType type = light->type();

		switch(type)
		{
		case Light::DirectionalLight:
			setLight(type, dircLightsCount, light.data());
			++dircLightsCount;
			break;
		case Light::PointLight:
			setLight(type, pointLightsCount, light.data());
			++pointLightsCount;
			break;
		case Light::SpotLight:
			setLight(type, spotLightsCount, light.data());
			++spotLightsCount;
			break;
		default:
			break;
		}
	}

	m_shaderProgram->setUniformValue("gNumDirectionalLights", dircLightsCount);
	m_shaderProgram->setUniformValue("gNumPointLights", pointLightsCount);
	m_shaderProgram->setUniformValue("gNumSpotLights", spotLightsCount);
}

void ShadingTechnique::setLight( Light::LightType type, uint index, const Light* light )
{
	QString lightString;

	switch(type)
	{
	case Light::DirectionalLight:
		lightString = "gDirectionalLights["+ QString::number(index) + "].";

		m_shaderProgram->setUniformValue((lightString + "Base.Color").toStdString().c_str(), light->color());
		m_shaderProgram->setUniformValue((lightString + "Base.Intensity").toStdString().c_str(), light->intensity());
		m_shaderProgram->setUniformValue((lightString + "Direction").toStdString().c_str(), light->direction());
		break;

	case Light::PointLight:
		lightString = "gPointLights["+ QString::number(index) + "].";

 		m_shaderProgram->setUniformValue((lightString + "Base.Color").toStdString().c_str(), light->color());
 		m_shaderProgram->setUniformValue((lightString + "Base.Intensity").toStdString().c_str(), light->intensity());
 		m_shaderProgram->setUniformValue((lightString + "Position").toStdString().c_str(), light->position());
		m_shaderProgram->setUniformValue((lightString + "Atten.Constant").toStdString().c_str(), light->constantAttenuation());
		m_shaderProgram->setUniformValue((lightString + "Atten.Linear").toStdString().c_str(), light->linearAttenuation());
		m_shaderProgram->setUniformValue((lightString + "Atten.Exp").toStdString().c_str(), light->quadraticAttenuation());
		break;

	case Light::SpotLight:
		lightString = "gSpotLights["+ QString::number(index) + "].";

		m_shaderProgram->setUniformValue((lightString + "Base.Base.Color").toStdString().c_str(), light->color());
		m_shaderProgram->setUniformValue((lightString + "Base.Base.Intensity").toStdString().c_str(), light->intensity());
		m_shaderProgram->setUniformValue((lightString + "Base.Atten.Constant").toStdString().c_str(), light->constantAttenuation());
		m_shaderProgram->setUniformValue((lightString + "Base.Atten.Linear").toStdString().c_str(), light->linearAttenuation());
		m_shaderProgram->setUniformValue((lightString + "Base.Atten.Exp").toStdString().c_str(), light->quadraticAttenuation());
		m_shaderProgram->setUniformValue((lightString + "Base.Position").toStdString().c_str(), light->position());
		m_shaderProgram->setUniformValue((lightString + "Direction").toStdString().c_str(), light->direction());
		break;

	default:
		break;
	}
}

void ShadingTechnique::setMVPMatrix(const mat4& mvp)
{
	m_shaderProgram->setUniformValue("gWVP", mvp);
}

void ShadingTechnique::setModelMatrix(const mat4& model)
{
	m_shaderProgram->setUniformValue("gWorld", model);
}

void ShadingTechnique::setViewMatrix( const mat4& view )
{
	m_shaderProgram->setUniformValue("viewMatrix", view);
}

void ShadingTechnique::setColorTextureUnit(unsigned int TextureUnit)
{
	m_shaderProgram->setUniformValue("gColorMap", TextureUnit);
}

void ShadingTechnique::setShadowMapTextureUnit(unsigned int TextureUnit)
{
	m_shaderProgram->setUniformValue("gShadowMap", TextureUnit);
}

void ShadingTechnique::setNormalMapTextureUnit(unsigned int TextureUnit)
{
	m_shaderProgram->setUniformValue("gNormalMap", TextureUnit);
}

void ShadingTechnique::setEyeWorldPos(const vec3& EyeWorldPos)
{
	m_shaderProgram->setUniformValue("gEyeWorldPos", EyeWorldPos);
}

void ShadingTechnique::setBoneTransform(uint Index, const mat4& Transform)
{
    assert(Index < MAX_BONES);
	QString boneString = "gBones["+ QString::number(Index) + "]";
	m_shaderProgram->setUniformValue(boneString.toStdString().c_str(), Transform);
}

void ShadingTechnique::setMaterial( const Material* mat )
{
	m_shaderProgram->setUniformValue("material.Ka", mat->m_ambientColor);
	m_shaderProgram->setUniformValue("material.Kd", mat->m_diffuseColor);
	m_shaderProgram->setUniformValue("material.Ks", mat->m_specularColor);
	m_shaderProgram->setUniformValue("material.Ke", mat->m_emissiveColor);
	m_shaderProgram->setUniformValue("material.hasDiffuseMap", mat->m_hasDiffuseMap);
	m_shaderProgram->setUniformValue("material.hasNormalMap", mat->m_hasNormalMap);
	m_shaderProgram->setUniformValue("material.shininessStrength", mat->m_shininessStrength);
	m_shaderProgram->setUniformValue("material.shininess", mat->m_shininess);
	m_shaderProgram->setUniformValue("material.roughnessValue", mat->m_roughness);
	m_shaderProgram->setUniformValue("material.fresnelReflectance", mat->m_fresnelReflectance);
	m_shaderProgram->setUniformValue("material.reflectFactor", mat->m_reflectFactor);
	m_shaderProgram->setUniformValue("material.refractiveIndex", mat->m_refractiveIndex);
}

void ShadingTechnique::setMatAmbientColor( const QColor& col )
{
	m_shaderProgram->setUniformValue("material.Ka", col);
}

void ShadingTechnique::setMatDiffuseColor( const QColor& col )
{
	m_shaderProgram->setUniformValue("material.Kd", col);
}

void ShadingTechnique::setMatSpecularColor( const QColor& col )
{
	m_shaderProgram->setUniformValue("material.Ks", col);
}

void ShadingTechnique::setMatEmissiveColor( const QColor& col )
{
	m_shaderProgram->setUniformValue("material.Ke", col);
}

void ShadingTechnique::setMatSpecularIntensity(float intensity)
{
	m_shaderProgram->setUniformValue("material.shininessStrength", intensity);
}

void ShadingTechnique::setMatSpecularPower(float power)
{
	m_shaderProgram->setUniformValue("material.shininess", power);
}

void ShadingTechnique::setMatRoughnessValue( float val )
{
	m_shaderProgram->setUniformValue("material.roughnessValue", val);
}

void ShadingTechnique::setMatFresnelReflectance( float val )
{
	m_shaderProgram->setUniformValue("material.fresnelReflectance", val);
}

void ShadingTechnique::setMatReflectFactor( float val )
{
	m_shaderProgram->setUniformValue("material.reflectFactor", val);
}

void ShadingTechnique::setMatRefractiveIndex( float val )
{
	m_shaderProgram->setUniformValue("material.refractiveIndex", val);
}
