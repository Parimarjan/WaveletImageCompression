import "regent"

-- FIXME: Remove token dependencies
-- FIXME: Check if order actually matters in the outer while loop?
-- FIXME: Can x and y things be done completely independently? I don't think
-- so - but check it by modifiying c++ code.
--
-- FIXME: Because we have two partitions, should we give each num_parallelism number
-- of partitions, or just divide those in half?

-- FIXME: spmd style programming.
-- Differences:
--    He runs on all three channels (RGB) separately.
--

-- Helper modules to handle PNG files and command line arguments
local png        = require("png_util")
local WaveletConfig = require("wavelet_config")

-- Some C APIs
local c     = regentlib.c
--local sqrt  = regentlib.sqrt(double)
--local cmath = terralib.includec("math.h")
--local PI = cmath.M_PI
--local unistd = terralib.includec("unistd.h")

terra wait_for(x : int) return 1 end


-- Field space for pixels
-- Should be the only required field because we make changes in place.
fspace Pixel
{
  value      : uint8;    -- value pixel in 8-bit gray scale
}

task block_task(r_image : region(ispace(int2d), Pixel))
where
  reads writes(r_image)
do
  return 1
end

task initialize(r_image : region(ispace(int2d), Pixel),
                filename : rawstring)
where
  reads writes(r_image)
do
  png.read_png_file(filename,
                    __physical(r_image.value),
                    __fields(r_image.value),
                    r_image.bounds)
end

task liftX(r_image    : region(ispace(int2d), Pixel), 
          step : int32)
where
  reads writes(r_image.value)
do 
  c.printf("in liftX!!\n")
  var ts_start = c.legion_get_current_time_in_micros()
  
  -- Don't want this because we want the exact indices and not sizes.
  --var size_x :int32 = r_mini_image.bounds:size().x
  --var size_y :int32 = r_mini_image.bounds:size().y
  
  var start_x :int32 = r_image.bounds.lo.x
  var start_y :int32 = r_image.bounds.lo.y
  var end_x : int32 = r_image.bounds.hi.x 
  var end_y : int32 = r_image.bounds.hi.y
  
  for y = r_image.bounds.lo.y, end_y+1, step do
    
    var base =  start_x
    regentlib.assert(base == 0, "base should always start with 0 in liftX")
    var base1 = base - step
    var base2 = base + step
    
    -- FIXME: Add parallelism at this stage. These two loops are just doing
    -- reductions, especially after step = 2.
    
    var x : int32

    for x = step, end_x + 1 - step, step*2 do
      r_image[{base+x, y}].value -= (r_image[{base1+x, y}].value + r_image[{base2+x, y}].value)/2
      --if step == 2 then
        --c.printf("base+x = %d, base1+x = %d, base2+x = %d\n", base+x, base1+x, base2+x)
      --end
    end
    
    -- Edge condition.
    -- FIXME: try commenting this out? If x = end_x, then this will fuck things
    -- up.
    if (x < end_x + 1) then
      r_image[{base+x, y}].value -= r_image[{base1+x, y}].value 
    end
    
    for x = step*2, end_x + 1 - step, step*2 do
      --if step == 2 then
        --c.printf("base+x = %d, base1+x = %d, base2+x = %d\n", base+x, base1+x, base2+x)
      --end
      r_image[{base+x, y}].value += (r_image[{base1+x, y}].value + r_image[{base2+x, y}].value)/4 
    end
    
    -- FIXME:
    if (x < end_x + 1) then
      r_image[{base+x, y}].value += r_image[{base1+x, y}].value / 2 
    end
    
  end

  var ts_end = c.legion_get_current_time_in_micros()
  c.printf("Lift x took %.3f sec.\n", (ts_end - ts_start) * 1e-6)
  return 1
end

-- Vertical lifting. Slightly more complicated than Horizontal lifting.

task liftY(r_image    : region(ispace(int2d), Pixel), 
          step : int32)
where
  reads writes(r_image.value)
do 
  var ts_start = c.legion_get_current_time_in_micros()
   
  var y0 : int32 = r_image.bounds.lo.y
  regentlib.assert(y0 == 0, "y0 was not 0 in liftY")
  -- we will need to apply the transform in x0 - size_x range.
  var x0 : int32 = r_image.bounds.lo.x

  var y1 : int32 = r_image.bounds.hi.y
  var x1 : int32 = r_image.bounds.hi.x
  
  --c.printf("low bounds are: %d, %d\n", r_image.bounds.lo.x, r_image.bounds.lo.y)
  --c.printf("hi bounds are %d, %d\n", x1, y1)

  for y = step, y1, step*2 do
    var base :int2d = {x0, y0 + y}
    var c1base :int2d = {x0, y0 + y - step}
    var c2base :int2d = c1base
    
    if (y + step) < y1 then 
      c2base = {x0, y0 + y + step}
    end
      
    var x : int32

    -- Linear filtering
    -- If we did only a few columns at a time? 
    
    -- Because we have set base as x0, now we are just adding 0...width of
    -- current region to x0.
    for x = 0, x1-x0, step do
      r_image[{base.x + x, base.y}].value -= (r_image[c1base + {x,0}].value + r_image[c2base + {x,0}].value) / 2
      --c.printf("base, x: %d, y: %d; c1base, x: %d, y: %d; c2base, x: %d, y: %d;\n", base.x+x,base.y,c1base.x+x,c1base.y,c2base.x+x,c2base.y)
    end 
  end
  
  -- Do we really need a separate loop for this?
  for y = step*2, y1, step*2 do
    var base :int2d = {x0 , y0 + y}
    -- FIXME: Does this always stay within range?
    var g1base :int2d = {x0, y0 + y - step}
    var g2base :int2d = g1base 
    if (y + step) < y1 then 
      g2base = {x0, y0 + y + step}
    end
    
    -- Same reason as in the loop above.
    for x = 0, x1-x0, step do
      r_image[base + {x, 0}].value += (r_image[g1base + {x,0}].value + r_image[g2base + {x,0}].value) / 4
      --c.printf("base, x: %d, y: %d; c1base, x: %d, y: %d; c2base, x: %d, y: %d;\n", base.x+x,base.y,g1base.x+x,g1base.y,g2base.x+x,g2base.y)
    end 
  end

  var ts_end = c.legion_get_current_time_in_micros()
  c.printf("Lift y took %.3f sec.\n", (ts_end - ts_start) * 1e-6)
  -- spurious dependencies.
  return 1
end

task unliftX(r_image    : region(ispace(int2d), Pixel), 
          step : int32)
where
  reads writes(r_image.value)
do 
  var ts_start = c.legion_get_current_time_in_micros()
  
  var start_x : int32 = r_image.bounds.lo.x
  var start_y : int32 = r_image.bounds.lo.y
  var end_x : int32 = r_image.bounds.hi.x
  var end_y : int32 = r_image.bounds.hi.y

  for y = start_y, end_y+1, step do
    
  --for y = r_image.bounds.lo.y + 0, size_y, step do
    -- just x values will be affected by these here.
    var base = start_x
    var base1 = base - step
    var base2 = base + step
    
    var x : int32
    -- even loop?
    for x = step*2, end_x + 1 - step, step*2 do
      r_image[{base+x, y}].value -= (r_image[{base1+x, y}].value + r_image[{base2+x, y}].value)/4 
    end
    
    -- FIXME: Uncomment this or fix this somehow.
    if (x < end_x+1) then
      r_image[{base+x, y}].value -= r_image[{base1+x, y}].value / 2 
    end
    
    for x = step, end_x + 1 - step, step*2 do
      r_image[{base+x, y}].value += (r_image[{base1+x, y}].value + r_image[{base2+x, y}].value)/2
    end
    
    -- Edge condition.
    if (x < end_x+1) then
      r_image[{base+x, y}].value += r_image[{base1+x, y}].value 
    end    
  end

  var ts_end = c.legion_get_current_time_in_micros()
  c.printf("UnLift x took %.3f sec.\n", (ts_end - ts_start) * 1e-6)
  -- spurious dependencies.
  return 1
end

task unliftY(r_image    : region(ispace(int2d), Pixel), 
          step : int32)
where
  reads writes(r_image.value)
do 

  var ts_start = c.legion_get_current_time_in_micros()
   
  var y0 : int32 = r_image.bounds.lo.y
  var x0 : int32 = r_image.bounds.lo.x

  var y1 : int32 = r_image.bounds.hi.y
  var x1 : int32 = r_image.bounds.hi.x
  
  --c.printf("low bounds are: %d, %d\n", r_image.bounds.lo.x, r_image.bounds.lo.y)
  --c.printf("hi bounds are %d, %d\n", x1, y1)
  
  for y = step*2, y1, step*2 do

    -- exactly same as in lift step
    var base :int2d = {x0 , y0 + y}
    var g1base :int2d = {x0, y0 + y - step}
    var g2base :int2d = g1base 
    if (y + step) < y1 then 
      g2base = {x0, y0 + y + step}
    end

    for x = 0, x1-x0, step do
      r_image[base + {x, 0}].value -= ((r_image[g1base + {x,0}].value + r_image[g2base + {x,0}].value) / 4)
    end 
  end

  for y = step, y1, step*2 do 
    var base :int2d = {x0, y0 + y}
    var c1base :int2d = {x0, y0 + y - step}
    var c2base :int2d = c1base
    
    if (y + step) < y1 then 
      c2base = {x0, y0 + y + step}
    end
    
    for x = 0, x1-x0, step do
      r_image[{base.x + x, base.y}].value += (r_image[c1base + {x,0}].value + r_image[c2base + {x,0}].value) / 2
    end 
  end
  
  var ts_end = c.legion_get_current_time_in_micros()
  --c.printf("Lift y took %.3f sec.\n", (ts_end - ts_start) * 1e-6)
  -- spurious dependencies.
  return 1
end


task saveImage(r_image : region(ispace(int2d), Pixel),
                filename : rawstring)
where
  reads(r_image.value)
do

  png.write_png_file(filename,
                     __physical(r_image.value),
                     __fields(r_image.value),
                     r_image.bounds)

end

task checkImage(r_image : region(ispace(int2d), Pixel))
where
  reads(r_image.value)
do

  c.printf("dimensions are %d, %d\n", r_image.bounds.hi.x,r_image.bounds.hi.y)
  
  -- FIXME: workaround to get an assert condition.
  for e in r_image do
    if not (r_image[e].value == 0) then
      regentlib.assert(0 == 0, "pixel value was 0")
    end
  end

end

-- Modify this to salvage it somehow. Maybe write out to a file or something.
task printOutImage(r_image: region(ispace(int2d), Pixel))
where
  reads(r_image.value)
do
  var size_y = r_image.bounds.hi.y  
  var size_x = r_image.bounds.hi.x
  
  for y = 0, size_y, 10 do
    c.printf("\n")
    for x = 0, size_x, 10 do
      c.printf("%d  ", r_image[{x,y}].value) 
    end
  end
  c.printf("finished printing this image\n")
end

task fill_image(r_image : region(ispace(int2d), Pixel),
                r_mini_image : region(ispace(int2d), Pixel))
where 
  reads (r_mini_image.value), writes (r_image.value)
do
  var low_bounds :int2d = r_image.bounds.lo
  -- c.printf("low bounds, combined: %d, %d; image: %d, %d\n", low_bounds.x, low_bounds.y, r_mini_image.bounds.lo.x, r_mini_image.bounds.lo.y)
  -- c.printf("combined image bounds are : %d, %d. r_image bounds are: %d, %d\n", r_image.bounds.hi.x, r_image.bounds.hi.y, r_mini_image.bounds.hi.x, r_mini_image.bounds.hi.y)
   
  for p in r_mini_image do
      var index : int2d = low_bounds + p
      r_image[index].value = r_mini_image[p].value
  end
end

task sequential_fill_image(r_image : region(ispace(int2d), Pixel),
                r_mini_image : region(ispace(int2d), Pixel),
                edge    : int32)
where 
  reads (r_mini_image), writes (r_image)
do

  for i = 0, edge, 1 do
    var new_y :int32 = i*r_mini_image.bounds.hi.y
    for j = 0, edge, 1 do
       var new_x :int32 = j*r_mini_image.bounds.hi.x 
       -- Now, we can fill up the image r_combined with r_image stuff
       for pixel in r_mini_image do
         var index : int2d = {new_x, new_y} + pixel 
         r_image[index].value = r_mini_image[pixel].value
       end 
    end
  end
end

task parallel_fill_image(r_image : region(ispace(int2d), Pixel),
                r_mini_image : region(ispace(int2d), Pixel),
                edge    : int32)
where 
  reads (r_mini_image), writes (r_image)
do
  var low_bounds = r_image.bounds.lo
  var hi_bounds = r_image.bounds.hi

  -- c.printf("r_image low bounds = %d, %d; high bounds = %d,%d\n",
  -- low_bounds.x,low_bounds.y, hi_bounds.x, hi_bounds.y)

  var new_y :int32 = r_image.bounds.lo.y

  -- mini_image is the full image so size_x makes sense.
  var size_x :int32 = r_mini_image.bounds:size().x

  -- looping over for each x-edge.
  for j = 0, edge, 1 do
     var new_x :int32 = j*size_x
     -- Now, we can fill up the image r_combined with r_image stuff

     for pixel in r_mini_image do
       var index : int2d = {new_x, new_y} + pixel 
       r_image[index].value = r_mini_image[pixel].value
     end 
  end
  c.printf("exiting parallel fill image\n")
end

task toplevel()
    
  var ts_start = c.legion_get_current_time_in_micros()
  var config : WaveletConfig
  config:initialize_from_command()

  config.skip_save = true
  c.printf("num parallelism is %d\n", config.num_parallelism)
  c.printf("edges are %d\n", config.edge)
  
  var edge : int32 = config.edge
  var size_image = png.get_image_size(config.filename_image)
  
  -- Combined big image
  var size_combined_image :int2d = {edge*size_image.x, edge*size_image.y}

  var r_mini_image = region(ispace(int2d, size_image), Pixel) 
  initialize(r_mini_image, config.filename_image) 
  
  -- Now we have the combined image, and we want to fill it up with the same
  -- data as in the original r_image.
  
  -- Right now the region hasn't been given actual memory yet.
  var r_image = region(ispace(int2d, size_combined_image), Pixel) 
  
   sequential_fill_image(r_image, r_mini_image, edge)
  -- Parallel init attempt 2
  --var init_coloring = c.legion_domain_point_coloring_create()
  ---- At every step/color we move down by the height of the image.
  ---- It will be an edge by edge square of images, so go down edge times.
    
  --var size_mini_y = r_mini_image.bounds:size().y
  --regentlib.assert(size_mini_y == size_image.y, "sizes y not same!")
  --for i = 0, edge, 1 do
    --var start_y = i*size_mini_y    
    ---- var end_y = start_y + size_mini_y
    --var end_y = start_y + size_mini_y - 1
    --var end_x = size_combined_image.x - 1

    ---- FIXME: Assuming the last value is included when partitioning.
    --c.legion_domain_point_coloring_color_domain(init_coloring, [int1d] (i),rect2d {{0, start_y}, {end_x,end_y}})
  --end

  --var init_colors = ispace(int1d, edge)

  ---- FIXME: it would throw an error if this was not disjoint right?
  --var p_init_image = partition(disjoint, r_image, init_coloring, init_colors)
  --c.legion_domain_point_coloring_destroy(init_coloring)

  ---- Another alternative is to equal divide big image and call seq_fill_image
  ---- on that.
  --for c in init_colors do
    --parallel_fill_image(p_init_image[c], r_mini_image, edge)
  --end
 
  -- We don't really have to saveImage as long as we are checking the values
  -- are valid.

  if not config.skip_save then
   saveImage(r_image, 'original_combined.png')
  end

  var step : uint32 = 1
  
  -- Once we divide r_image into colors, can we still get size of each color
  -- partition? Also we want to divide it such that its a bunch of rows.
  -- Will we need the exact same partition while decompressing?
  
  var size_x : uint32 = r_image.bounds.hi.x
  var size_y : uint32 = r_image.bounds.hi.y
  
  -- Want to launch the while loop on each region separately.
  
  var x_parallelism = config.num_parallelism

  -- FIXME: Might need to adjust this to be the same size for spmd?
  var x_coloring = c.legion_domain_point_coloring_create()
  
  var chunk_precise = [float](size_y) / x_parallelism
  c.printf("chunk precise is %0.4f\n", chunk_precise)

  var chunk_height = size_y / x_parallelism
  c.printf("chunk height is %d\n", chunk_height)

  -- FIXME: Try to not include the final region at all? That should fix errors
  -- that go over the bound.
  --
  
  for i = 0, x_parallelism, 1 do
    var start_y = i*chunk_height
    var end_y = start_y + chunk_height - 1
    
    -- FIXME: maybe I shouldn't be doing this?
    --if i == x_parallelism-1 then
      --end_y = size_y
    --end

    c.legion_domain_point_coloring_color_domain(x_coloring, [int1d] (i),rect2d {{0, start_y}, {size_x,end_y}})
  end

  var x_colors = ispace(int1d, x_parallelism)
  var p_x_combined_image = partition(disjoint, r_image, x_coloring, x_colors)
  c.legion_domain_point_coloring_destroy(x_coloring)
  
  var y_parallelism = config.num_parallelism
  var y_coloring = c.legion_domain_point_coloring_create()

  var precise_chunk = [float](size_x) / y_parallelism
  c.printf("precise chunk is %0.4f\n", precise_chunk)
  var chunk_width = size_x / y_parallelism
  c.printf("chunk width is %d\n", chunk_width)
  
  for i = 0, y_parallelism, 1 do
    var start_x = i*chunk_width
    var end_x = start_x + chunk_width - 1

    --if i == y_parallelism-1 then
      --end_x = size_x
    --end
    --c.printf("start_x = %d, end_x = %d\n", start_x, end_x)
    c.legion_domain_point_coloring_color_domain(y_coloring, [int1d](i), rect2d{{start_x, 0}, {end_x,size_y}})
  end
  var y_colors = ispace(int1d, y_parallelism) 
  var p_y_combined_image = partition(disjoint, r_image, y_coloring, y_colors)
  c.legion_domain_point_coloring_destroy(y_coloring)
  
  var token = 0
  
  -- FIXME: This is just an arbitrary limit being put on step.
  while (step < size_x/2 or step < size_y/2) do
    if (step < size_x/2) then
      -- Can I do partition.colors? why not?
      for i = 0, x_parallelism, 1 do 
         liftX(p_x_combined_image[i], step)
      end
    end

    if (step < size_y) then
      for i = 0, y_parallelism, 1 do
       liftY(p_y_combined_image[i], step)
      end
    end
    step = step*2
  end
  
  if not config.skip_save then
   saveImage(r_image, 'lifted_combined.png')
  end

  -- Let's do the unlifting to check if it works 
  -- Gets step to the highest value so we can run it in reverse....
    
  while (step*2 < size_x or step*2 < size_y) do 
    step *= 2
  end

  while (step >=1) do
    if (step < size_y) then -- vertical lifting 
      for i = 0, y_parallelism, 1 do
        unliftY(p_y_combined_image[i], step)
      end
    end

    if (step < size_x) then
      for i = 0, x_parallelism, 1 do 
        unliftX(p_x_combined_image[i], step)
      end
    end
    step /= 2
  end
    
   
  -- whenever I call this then full r_image is being processed by each node I
  -- I think? So that would definitely fuck up the sizes and stuff.
  -- checkImage(r_image)
  -- FIXME: saveImage if we need to present it.
  
  -- Can I run this in parallel too?
  if not config.skip_save then
    saveImage(r_image, 'unlifted_combined.png') 
  end
  
  for i = 0, y_parallelism, 1 do
   token += block_task(p_y_combined_image[i])
  end

  for i = 0, x_parallelism do
    token += block_task(p_x_combined_image[i])
  end

  wait_for(token)
  var ts_end = c.legion_get_current_time_in_micros()
  c.printf("Total time: %.6f sec.\n", (ts_end - ts_start) * 1e-6)
end

regentlib.start(toplevel)
