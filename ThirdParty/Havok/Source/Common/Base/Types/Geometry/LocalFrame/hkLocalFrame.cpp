/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Types/Geometry/LocalFrame/hkLocalFrame.h>

void hkLocalFrame::getTransformToRoot( hkTransform& transform ) const
{
	const hkLocalFrame* parentFrame = getParentFrame();

	if ( parentFrame )
	{
		hkTransform rootFromParent;

		// recur to the parent
		getParentFrame()->getTransformToRoot( rootFromParent );

		hkTransform parentFromMe;

		getLocalTransform( parentFromMe );

		transform.setMul( rootFromParent, parentFromMe );
	}
	else
	{
		getLocalTransform( transform );
	}
}

void hkLocalFrame::getPositionInRoot( hkVector4& position ) const
{
	getLocalPosition( position );

	const hkLocalFrame* parentFrame = getParentFrame();

	while( parentFrame != HK_NULL )
	{
		hkTransform parentTransform;
		parentFrame->getLocalTransform( parentTransform );
		position.setTransformedPos( parentTransform, position );
		parentFrame = parentFrame->getParentFrame();
	}
}

void hkLocalFrame::getLocalPosition( hkVector4& position ) const
{
	hkTransform t;

	// get the transform for this local frame
	getLocalTransform( t );

	// get the position from the transform
	position = t.getTranslation();
}

void hkLocalFrame::getDescendants( hkArrayBase<const hkLocalFrame*>& descendants, hkMemoryAllocator& a) const
{
	if( getNumChildFrames() == 0)
	{
		return;
	}
	else
	{
		for( int i = 0; i < getNumChildFrames(); ++i )
		{
			hkLocalFrame* child = getChildFrame(i);

			if( child != HK_NULL )
			{
				descendants._pushBack(a, child);
				child->getDescendants(descendants, a);
			}			
		}
	}
}

hkSimpleLocalFrame::~hkSimpleLocalFrame()
{
	const int count = m_children.getSize();

	for( int i = 0; i < count; i++ )
	{
		m_children[i]->removeReference();
	}

	if ( m_group != HK_NULL )
	{
		m_group->removeReference();
	}
}

void hkSimpleLocalFrame::getLocalTransform( hkTransform& transform ) const
{
	transform = m_transform;
}

void hkSimpleLocalFrame::setLocalTransform( const hkTransform& transform )
{
	m_transform = transform;
}

void hkSimpleLocalFrame::getLocalPosition( hkVector4& position ) const
{
	position = m_transform.getTranslation();
}

void hkSimpleLocalFrame::getNearbyFrames( const hkVector4& target, hkReal maxDistance, hkLocalFrameCollector& collector ) const
{
	hkReal distance = target.distanceTo( m_transform.getTranslation() ).getReal();

	// collect the frame if it is within range of the desired position
	if ( distance <= maxDistance )
	{
		// add the frame to the collector
		collector.addFrame( this, distance );
	}

	const int numChildren = m_children.getSize();

	if ( numChildren > 0 )
	{
		hkVector4 localTarget;
		localTarget.setTransformedInversePos( m_transform, target );

		for( int i = 0; i < m_children.getSize(); i++ )
		{
			// recur
			m_children[i]->getNearbyFrames( localTarget, maxDistance, collector );
		}
	}
}

void hkSimpleLocalFrame::setGroup( const hkLocalFrameGroup* group )
{
	if ( group != HK_NULL )
	{
		group->addReference();
	}

	if ( m_group != HK_NULL )
	{
		m_group->removeReference();
	}

	m_group = group;
}


int hkSimpleLocalFrame::getNumChildFrames() const
{
	return m_children.getSize();
}

hkLocalFrame* hkSimpleLocalFrame::getChildFrame( int i ) const
{
	return m_children[ i ];
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
