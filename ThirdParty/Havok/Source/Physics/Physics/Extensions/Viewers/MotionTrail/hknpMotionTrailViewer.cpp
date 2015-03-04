/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Extensions/Viewers/MotionTrail/hknpMotionTrailViewer.h>

#include <Common/Visualize/hkProcessFactory.h>
#include <Common/Base/Types/Color/hkColor.h>


int hknpMotionTrailViewer::s_tag = 0;

void HK_CALL hknpMotionTrailViewer::registerViewer( hkProcessFactory& factory )
{
	s_tag = factory.registerProcess( getName(), create );
}

hkProcess* HK_CALL hknpMotionTrailViewer::create( const hkArray<hkProcessContext*>& contexts )
{
	return new hknpMotionTrailViewer( contexts );
}


hknpMotionTrailViewer::hknpMotionTrailViewer( const hkArray<hkProcessContext*>& contexts )
: hknpViewer( contexts )
{
	m_skip = 0;
}

void hknpMotionTrailViewer::step( hkReal deltaTime )
{
	if( !m_context )
	{
		return;
	}

	HK_TIMER_BEGIN( "MotionTrailViewer", this );

	int max = 32;

	// Update
	//if ( m_skip++ == 3 )
	{
		m_skip = 0;

		for( int wi=0; wi<m_context->getNumWorlds(); wi++ )
		{
			const hknpMotionManager& motionManager = m_context->getWorld(wi)->m_motionManager;
			m_histories.setSize( motionManager.getPeakMotionId().value() + 1 );
			for( hknpMotionIterator it(motionManager); it.isValid(); it.next() )
			{
				const hknpMotion& motion = it.getMotion();
				if( motion.isActive() )
				{
					hkVector4 com = motion.getCenterOfMassInWorld();

					History& history = m_histories[ it.getMotionId().value() ];
					if( !history.m_positions.isEmpty() )
					{
						int size = history.m_positions.getSize();
						int lastIndex = (history.m_index + size - 1) % max;
						int newIndex = (history.m_index + size) % max;

						hkVector4 last = history.m_positions[lastIndex];

						//if (last.distanceTo(curr) >= hkSimdReal::fromFloat(0.1f))
						{
							if (size >= max)
							{
								history.m_positions[newIndex] = com;
								history.m_index = newIndex+1;
							}
							else
							{
								history.m_positions.pushBack(com);
							}
						}
					}
					else
					{
						history.m_index = 0;
						history.m_positions.pushBack(com);
					}
				}
			}
		}
	}

	// Draw
	for( int wi=0; wi<m_context->getNumWorlds(); wi++ )
	{
		const hknpMotionManager& motionManager = m_context->getWorld(wi)->m_motionManager;
		m_histories.setSize( motionManager.getPeakMotionId().value() + 1 );
		for( hknpMotionIterator it(motionManager); it.isValid(); it.next() )
		{
			const hknpMotion& motion = it.getMotion();
			History& history = m_histories[ it.getMotionId().value() ];
			if( motion.isActive() )
			{
				for( int i = history.m_index, c = 0; c+1 < history.m_positions.getSize(); c++, i++)
				{
					m_displayHandler->displayLine( history.m_positions[(i)%max], history.m_positions[(i+1)%max], 0xffff66ff, 0, s_tag );
				}
			}
			else
			{
				history.m_positions.clear();
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
