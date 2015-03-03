/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Extensions/LeafShapeSensor/hknpLeafShapeSensor.h>

#include <Common/Base/Container/PointerMap/hkMap.cxx>
#include <Common/Base/Types/Geometry/Aabb/hkAabbUtil.h>

#include <Physics/Physics/Collide/Shape/hknpShapeQueryInterface.h>
#include <Physics/Physics/Collide/Shape/Composite/hknpCompositeShape.h>


hknpLeafShapeSensor::hknpLeafShapeSensor( hknpWorld* world, hkInt32 capacity )
	: m_capacity(capacity), m_world(world), m_aabbExpansionMargin(0.0f)
{
	HK_WARN_ON_DEBUG_IF((capacity > 256), 0x1ce93563, "hknpLeafShapeSensor cannot collect more than 256 child physics shapes.");
	m_capacity = (capacity > 256)? 256 : capacity;
}

hknpLeafShapeSensor::~hknpLeafShapeSensor()
{
	// Clean up registered bodies (if any)
	hkAabb aabb; aabb.setEmpty();
	clearSet(m_bodyContexts, aabb, *this, &hknpLeafShapeSensor::bodyCleaup );
}

template < typename KEY, typename VALUE, typename OPS, typename CONTEXT, typename TYPE, typename CALLBACK3 >
HK_FORCE_INLINE void hknpLeafShapeSensor::clearSet(	hkMap<KEY, hknpLeafShapeSensor::Data<VALUE>,OPS>& keys, CONTEXT& context, TYPE& object, const CALLBACK3 leaveCallback )
{
	// Go through Keys and signal leaving keys, passing the user data
	typename hkMap<KEY,VALUE,OPS>::Iterator iterator = keys.getIterator();
	while (keys.isValid(iterator))
	{
		Data<VALUE> value = keys.getValue(iterator);

		KEY key = keys.getKey(iterator);

		// Call user leave callback
		(object.*(leaveCallback))(context, key, value.m_userData );

		iterator = keys.getNext(iterator);
	}

	keys.clear();
}

template < typename KEY, typename VALUE, typename OPS,
			typename ARRAY, typename CONTEXT, typename TYPE,
				typename CALLBACK1, typename CALLBACK2, typename CALLBACK3 >
HK_FORCE_INLINE void hknpLeafShapeSensor::updateSet( hkMap<KEY,hknpLeafShapeSensor::Data<VALUE>,OPS>& keys, ARRAY& observedKeys, CONTEXT& context, TYPE& object,
														const CALLBACK1 enterCallback, const CALLBACK2 stayCallback, const CALLBACK3 leaveCallback )
{
	// Mark all Keys as leaving
	typename hkMap<KEY,VALUE,OPS>::Iterator iterator = keys.getIterator();
	while (keys.isValid(iterator))
	{
		hknpLeafShapeSensor::Data<VALUE> value = keys.getValue(iterator);

		// Set callback type to leaving
		{
			value.m_callbackType = LEAVING;
		}

		keys.setValue(iterator, value);

		// Next key
		iterator = keys.getNext(iterator);
	}

	// Mark matching keys as STAYING, new keys as ENTERING
	for (int i=0; i<observedKeys.getSize(); i++)
	{
		KEY key = observedKeys[i];

		if ( keys.hasKey( key ) )
		{
			typename hkMap<KEY,VALUE>::Iterator iter = keys.findKey(key);

			// Grab value data
			Data<VALUE> value = keys.getValue(iter);

			// Set callback to stay
			value.m_callbackType = STAYING;

			// Call user stay callback
			VALUE userData = (object.*stayCallback)(context, key, value.m_userData );

			// User data will be 0 if the body is no longer registered
			if(userData == 0)
			{
				value.m_callbackType = LEAVING;
			}

			// Already present keys stays
			keys.setValue( iter, value );
		}
		else
		{
			// Call user enter callback, userData will be 0 if body is not registered or filtered
			VALUE userData = (object.*enterCallback)(context, key);

			if(userData != 0)
			{
				// Set value for map entry
				Data<VALUE> data;
				{
					data.m_userData = userData;
					data.m_callbackType = ENTERING;
				}
				// New keys are entering
				keys.insert( key, data);
			}
		}
	}

	hkArray<KEY> keysToRemove;
	// Go through Keys and signal leaving keys, passing the user data
	iterator = keys.getIterator();
	while (keys.isValid(iterator))
	{
		Data<VALUE> value = keys.getValue(iterator);

		if( value.m_callbackType == LEAVING )
		{
			KEY key = keys.getKey(iterator);

			// Call user leave callback
			(object.*(leaveCallback))(context, key, value.m_userData );

			keysToRemove.pushBack(key);
		}

		iterator = keys.getNext(iterator);
	}

	for(int kIdx=0; kIdx<keysToRemove.getSize(); kIdx++)
	{
		keys.remove(keysToRemove[kIdx]);
	}
}

HK_FORCE_INLINE hknpLeafShapeSensor::Context* hknpLeafShapeSensor::bodyEntered( const hkAabb& aabb, const hknpBodyId& bodyId )
{
	// Filter
	if(!isCollisionEnabled(bodyId))
	{
		return HK_NULL;
	}

	return createContext( aabb, bodyId );
}

HK_FORCE_INLINE void hknpLeafShapeSensor::bodyLeft( const hkAabb& aabb, const hknpBodyId& bodyId, Context* const context )
{
	if( context )
	{
		removeContext( context );
	}
}

HK_FORCE_INLINE void hknpLeafShapeSensor::bodyCleaup( const hkAabb& aabb, const hknpBodyId& bodyId, Context* const context )
{
	if( context )
	{
		delete context;
	}
}

HK_FORCE_INLINE hknpLeafShapeSensor::Context* hknpLeafShapeSensor::bodyStayed( const hkAabb& aabb, const hknpBodyId& bodyId, hknpLeafShapeSensor::Context* const context )
{
	// if interested in this body (i.e., registered and not filtered)
	if( isCollisionEnabled(bodyId) && context )
	{
		processContext( aabb, context );
		return context;
	}
	else if(context)
	{
		// destroy context if it exists and we are no longer interested in this body (firing shape left callbacks)
		return HK_NULL;
	}

	return HK_NULL;
}

void hknpLeafShapeSensor::setAabbExpansionMargin( hkReal margin )
{
	m_aabbExpansionMargin = margin;
}

void hknpLeafShapeSensor::setCapacity( hkInt32 capacity )
{
	HK_WARN_ON_DEBUG_IF((capacity > 256), 0x1ce93563, "hknpLeafShapeSensor cannot collect more than 256 child physics shapes.");
	m_capacity = (capacity > 256)? 256 : capacity;
}

void hknpLeafShapeSensor::updateAabb( hkAabb& aabb )
{
	HK_TIMER_BEGIN("hknpLeafShapeSensor::updateAabb", HK_NULL);

	// Query broad-phase AABB
	hkArray<hknpBodyId> overlaps;
	m_world->queryAabb(aabb, overlaps);

	// Fire callbacks for bodies
	updateSet(m_bodyContexts, overlaps, aabb, *this,
		&hknpLeafShapeSensor::bodyEntered,
		&hknpLeafShapeSensor::bodyStayed,
		&hknpLeafShapeSensor::bodyLeft );

	// Updated callback
	shapesUpdatedCallback();

	HK_TIMER_END();
}

hkUlong hknpLeafShapeSensor::getUserData( const LeafShapeId& id )
{
	// Get context
	Context* context = m_bodyContexts.getValue( m_bodyContexts.findKey(id.m_bodyId) ).m_userData;

	if(!context->m_userData)
	{
		Data<hkUlong> value = context->m_shapeKeys->getValue( context->m_shapeKeys->findKey(id.m_shapeKey) );
		return value.m_userData;
	}

	return context->m_userData;
}

hknpLeafShapeSensor::Context* hknpLeafShapeSensor::createContext( const hkAabb& aabb, const hknpBodyId& bodyId )
{
	const hknpBody& body = m_world->getBody(bodyId);
	bool collideWithBody = collideWithBodyOnly(bodyId);

	// Setup a new context for this body
	Context* context = new Context( bodyId, body.isStatic(), (body.m_shape->m_dispatchType == hknpCollisionDispatchType::COMPOSITE), collideWithBody );

	if( collideWithBody )
	{
		context->m_userData = bodyEnteredCallback( LeafShapeId(bodyId, HKNP_INVALID_SHAPE_KEY) );
	}
	else
	{
		if ( context->m_isComposite )
		{
			processContextComposite(aabb, context);
		}
		else
		{
			context->m_userData = shapeEnteredCallback( LeafShapeId(bodyId, HKNP_INVALID_SHAPE_KEY) );
		}
	}

	return context;
}

void hknpLeafShapeSensor::processContext( const hkAabb& aabb, Context* const context )
{
	if( context->m_collideWithBodyOnly )
	{
		bodyUpdatedCallback( LeafShapeId(context->m_bodyId, HKNP_INVALID_SHAPE_KEY), context->m_userData );
	}
	else
	{
		if ( context->m_isComposite )
		{
			processContextComposite(aabb, context);
		}
		else
		{
			shapeUpdatedCallback( LeafShapeId(context->m_bodyId, HKNP_INVALID_SHAPE_KEY), context->m_userData );
		}
	}
}

void hknpLeafShapeSensor::processContextComposite( const hkAabb& aabb, Context* context )
{
	const hknpBody& body = m_world->getBody(context->m_bodyId);

	if(body.isStatic())
	{
		// Early out if new AABB is still within expanded AABB
		hkAabb landscapeAabb; getAaab(true, landscapeAabb);
		if (context->m_aabb.contains(landscapeAabb))
		{
			return;
		}

		// Update the expanded AABB.
		// The expansion is just a heuristic, its done to make sure that the AABB is only re-queried every time the
		// current AABB breaks out of the expanded one (the expanded one persists from last query)
		{
			context->m_aabb = landscapeAabb;
			context->m_aabb.expandBy( hkSimdReal::fromFloat(m_aabbExpansionMargin) );
		}
	}
	else
	{
		context->m_aabb = aabb;
	}

	// Query the composite shape for shape keys overlapping this AABB
	hkInplaceArray<hknpShapeKey, 256> returnedKeys;
	{
		const hknpCompositeShape* composite = static_cast<const hknpCompositeShape*>(body.m_shape);
		hknpCollisionQueryContext queryContext(HK_NULL, HK_NULL);
		hknpAabbQuery aabbQuery(context->m_aabb); hknpShapeQueryInfo info; info.setFromBody(body);
		hknpQueryFilterData filterData;
		hkAabbUtil::transformAabbIntoLocalSpace(body.getTransform(), context->m_aabb, aabbQuery.m_aabb);
		hknpShapeQueryInterface::queryAabb(&queryContext, aabbQuery, *composite, filterData, info, &returnedKeys );
	}

	// As queryAabb does not cap the return result, we cap it here
	
	
	if (returnedKeys.getSize() > m_capacity)
	{
		returnedKeys.setSize(m_capacity);
	}

	// update set of shape keys
	updateSet( *context->m_shapeKeys, returnedKeys, context, *this,
		&hknpLeafShapeSensor::childShapeEntered,
		&hknpLeafShapeSensor::childShapeStayed,
		&hknpLeafShapeSensor::childShapeLeft );

	returnedKeys.clearAndDeallocate();
}

void hknpLeafShapeSensor::removeContext( Context* context )
{
	if( context->m_collideWithBodyOnly)
	{
		bodyLeftCallback( LeafShapeId(context->m_bodyId, HKNP_INVALID_SHAPE_KEY), context->m_userData );
	}
	else
	{
		if ( context->m_isComposite)
		{
			// All keys will be leaving
			clearSet( *context->m_shapeKeys, context, *this, &hknpLeafShapeSensor::childShapeLeft );
		}
		else
		{
			// Report the non-composite shape leaving
			shapeLeftCallback( LeafShapeId(context->m_bodyId, HKNP_INVALID_SHAPE_KEY), context->m_userData );
		}
	}
	// Clean up
	delete context;
}

void hknpLeafShapeSensor::removeContext(const hknpBodyId& bodyId)
{
	hkMap<hknpBodyId,Data<Context*> >::Iterator it = m_bodyContexts.findKey(bodyId);

	if( !m_bodyContexts.isValid(it) )
		return;

	Context* context = m_bodyContexts.getValue(it).m_userData;
	removeContext(context);
	m_bodyContexts.remove(bodyId);
}

HK_FORCE_INLINE hkUlong hknpLeafShapeSensor::childShapeEntered( const Context* const context, const hknpShapeKey& shapeKey )
{
	// Notify user
	return shapeEnteredCallback( LeafShapeId(context->m_bodyId, shapeKey) );
}

HK_FORCE_INLINE void hknpLeafShapeSensor::childShapeLeft( const Context* const context, const hknpShapeKey& shapeKey, hkUlong user )
{
	// Notify user
	shapeLeftCallback( LeafShapeId(context->m_bodyId, shapeKey), user );
}

HK_FORCE_INLINE hkUlong hknpLeafShapeSensor::childShapeStayed( const Context* const context, const hknpShapeKey& shapeKey, hkUlong user )
{
	// Notify user
	shapeUpdatedCallback( LeafShapeId(context->m_bodyId, shapeKey), user );

	return user;
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
