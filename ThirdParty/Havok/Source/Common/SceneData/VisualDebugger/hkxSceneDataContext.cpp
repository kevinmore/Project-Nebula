/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/SceneData/hkSceneData.h>
#include <Common/SceneData/VisualDebugger/hkxSceneDataContext.h>
#include <Common/SceneData/VisualDebugger/Viewer/hkxSceneViewer.h>

hkxSceneDataContext::hkxSceneDataContext() :
	m_allowMipmap(true)
{
}

hkxSceneDataContext::~hkxSceneDataContext()
{
	for( hkInt32 i = m_scenes.getSize() - 1; i >= 0; --i )
	{
		removeScene(m_scenes[i]);
	}
}

void HK_CALL hkxSceneDataContext::registerAllSceneDataViewers()
{	
	hkxSceneViewer::registerViewer();
}

const char* hkxSceneDataContext::getType()
{
	return HK_SCENE_DATA_CONTEXT_TYPE_STRING;
}

void hkxSceneDataContext::addScene( hkxScene* scene )
{
	HK_ASSERT2(0xd8f81a9, m_scenes.indexOf( scene ) == -1, "You tried to add a scene that is already added to the context.");

	m_scenes.pushBack(scene);

	for( hkInt32 i = 0; i < m_listeners.getSize(); ++i )
	{
		m_listeners[i]->sceneAddedCallback( scene );
	}
}

void hkxSceneDataContext::removeScene( hkxScene* scene )
{
	hkInt32 index = m_scenes.indexOf(scene);

	HK_ASSERT2(0x34e822fb, index != -1, "You tried to remove a scene that was not in the context." );

	if( index == -1 )
	{
		return;
	}

	m_scenes.removeAt(index);		

	for( hkInt32 i = 0; i < m_listeners.getSize(); ++i )
	{
		m_listeners[i]->sceneRemovedCallback( scene );
	}	
}

const hkArray<hkxScene*>& hkxSceneDataContext::getScenes() const
{
	return m_scenes;
}

void hkxSceneDataContext::addListener( hkxSceneDataContextListener* listener )
{
	HK_ASSERT2(0x694c52fb, m_listeners.indexOf( listener ) == -1, "You tried to add a listener that is already added to the context.");

	m_listeners.pushBack(listener);	
}

void hkxSceneDataContext::removeListener( hkxSceneDataContextListener* listener )
{
	hkInt32 index = m_listeners.indexOf(listener);

	HK_ASSERT2(0x6b3aa5cf, index != -1, "You tried to remove a listener that was not in the context." );

	m_listeners.removeAt(index);	
}

void hkxSceneDataContext::addTextureSearchPath(const char* path)
{
	m_searchPaths.pushBack( path );
}

void hkxSceneDataContext::clearTextureSearchPaths()
{
	m_searchPaths.clear();
}

const hkArray<const char*>& hkxSceneDataContext::getTextureSearchPaths() const
{
	return m_searchPaths;
}

void hkxSceneDataContext::setAllowTextureMipmap( hkBool on )
{
	m_allowMipmap = on;
}

bool hkxSceneDataContext::getAllowTextureMipmap()
{
	return m_allowMipmap;
}

/*
 * Havok SDK - Base file, BUILD(#20130912)
 * 
 * Confidential Information of Havok.  (C) Copyright 1999-2013
 * Telekinesys Research Limited t/a Havok. All Rights Reserved. The Havok
 * Logo, and the Havok buzzsaw logo are trademarks of Havok.  Title, ownership
 * rights, and intellectual property rights in the Havok software remain in
 * Havok and/or its suppliers.
 * 
 * Use of this software for evaluation purposes is subject to and indicates
 * acceptance of the End User licence Agreement for this product. A copy of
 * the license is included with this software and is also available from salesteam@havok.com.
 * 
 */
