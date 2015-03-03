/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Physics/Physics/Extensions/Viewers/hknpViewer.h>


hknpViewer::hknpViewer( const hkArray<hkProcessContext*>& contexts )
:	hkProcess( true ),	// all selectable
	m_context( HK_NULL ),
	m_selectedBody( hknpBodyId(0) ),
	m_worldForViewerSpecificBody( HK_NULL )
{
	// Find our context
	const int numContexts = contexts.getSize();
	for ( int i=0; i < numContexts; ++i )
	{
		if ( hkString::strCmp( HKNP_PROCESS_CONTEXT_TYPE_STRING, contexts[i]->getType() ) == 0 )
		{
			m_context = static_cast<hknpProcessContext*>( contexts[i] );
			break;
		}
	}

	if ( m_context )
	{
		m_context->addWorldListener(this); // context is a world deletion listener and will pass it on
		m_context->addReference(); // so that it can't be deleted before us.
	}
}

void hknpViewer::init()
{
	if ( m_context )
	{
		// Add any worlds that exist in the context
		for ( int i=0; i < m_context->getNumWorlds(); ++i )
		{
			worldAddedCallback( m_context->getWorld(i) );
		}
	}
}

hknpViewer::~hknpViewer()
{
	if ( m_context )
	{
		m_context->removeWorldListener(this);
		m_context->removeReference(); // let it go.
		m_context = HK_NULL;
	}
}


/*static*/ void HK_CALL hknpViewer::displayOrientedPoint(hkDebugDisplayHandler* displayHandler, hkVector4Parameter position, const hkRotation& rot, hkSimdRealParameter size, hkColor::Argb color, int id, int tag)
{
	hkVector4 p1, p2, scaled;

	const hkVector4& x = rot.getColumn(0);
	const hkVector4& y = rot.getColumn(1);
	const hkVector4& z = rot.getColumn(2);

	scaled.setMul(size,x);
	p1.setSub(position,scaled);
	scaled.setMul(-size,x);
	p2.setSub(position,scaled);
	displayHandler->displayLine(p1, p2, color, id, tag);

	scaled.setMul(size,y);
	p1.setSub(position,scaled);
	scaled.setMul(-size,y);
	p2.setSub(position,scaled);
	displayHandler->displayLine(p1, p2, color, id, tag);

	scaled.setMul(size,z);
	p1.setSub(position,scaled);
	scaled.setMul(-size,z);
	p2.setSub(position,scaled);
	displayHandler->displayLine(p1, p2, color, id, tag);
}

/*static*/ void HK_CALL hknpViewer::displayArrow(hkDebugDisplayHandler* displayHandler, hkVector4Parameter startPos, hkVector4Parameter arrowDirection, hkVector4Parameter perpDirection, hkColor::Argb color, hkSimdRealParameter scale, int id, int tag)
{
	hkVector4 endPos = startPos;
	endPos.addMul(scale, arrowDirection);
	displayHandler->displayLine(startPos, endPos, color, id, tag);

	hkQuaternion FortyFiveAboutPerpDirection; FortyFiveAboutPerpDirection.setAxisAngle(perpDirection, HK_REAL_PI * 0.25f);
	hkQuaternion MinusNinetyAboutPerpDirection; MinusNinetyAboutPerpDirection.setAxisAngle(perpDirection, HK_REAL_PI * -0.5f);

	hkVector4 headDirection = arrowDirection;
	headDirection._setRotatedDir(FortyFiveAboutPerpDirection, headDirection);
	hkVector4 temp = endPos;
	temp.addMul(-scale * hkSimdReal::getConstant(HK_QUADREAL_INV_3), headDirection);
	displayHandler->displayLine(endPos, temp, color, id, tag);

	headDirection._setRotatedDir(MinusNinetyAboutPerpDirection, headDirection);
	temp = endPos;
	temp.addMul(-scale * hkSimdReal::getConstant(HK_QUADREAL_INV_3), headDirection);
	displayHandler->displayLine(endPos, temp, color, id, tag);
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
