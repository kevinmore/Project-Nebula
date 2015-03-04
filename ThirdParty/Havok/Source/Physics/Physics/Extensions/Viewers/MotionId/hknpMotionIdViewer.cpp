/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Extensions/Viewers/MotionId/hknpMotionIdViewer.h>

#include <Common/Visualize/hkDebugDisplayHandler.h>
#include <Common/Visualize/hkProcessFactory.h>
#include <Common/Visualize/hkDebugDisplay.h>


int hknpMotionIdViewer::s_tag = 0;

void HK_CALL hknpMotionIdViewer::registerViewer( hkProcessFactory& factory )
{
	s_tag = factory.registerProcess( getName(), create );
}

hkProcess* HK_CALL hknpMotionIdViewer::create( const hkArray<hkProcessContext*>& contexts )
{
	return new hknpMotionIdViewer( contexts );
}


hknpMotionIdViewer::hknpMotionIdViewer( const hkArray<hkProcessContext*>& contexts )
:	hknpViewer( contexts )
{
}

void hknpMotionIdViewer::step( hkReal deltaTime )
{
	if( !m_context )
	{
		return;
	}

	HK_TIMER_BEGIN( "MotionIdViewer", this );

	for( int wi=0; wi<m_context->getNumWorlds(); wi++ )
	{
		const hknpMotionManager& motionManager = m_context->getWorld(wi)->m_motionManager;
		for( hknpMotionIterator it = motionManager.getMotionIterator(); it.isValid(); it.next() )
		{
			const hknpMotion& motion = it.getMotion();
			if( motion.isActive() )
			{
				char str[256];
				hkString::sprintf( str, "%i", it.getMotionId().value() );
				m_displayHandler->display3dText( str, motion.getCenterOfMassInWorld(), hkColor::MAGENTA, 0, s_tag );
			}
		}
	}

	HK_TIMER_END();
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
